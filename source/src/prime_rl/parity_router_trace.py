from __future__ import annotations

import base64
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import re
import secrets
import stat
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


trace_schema_version = 1
trace_dir_env = "PRIME_PARITY_ROUTER_TRACE_DIR"
session_name = "session.json"
runtime_marker_name = "h200-runtime.json"
trainer_provenance_name = "trainer-runtime.json"
router_debug_pattern = re.compile(
    r"^(trainer|h200)-rank-(\d+)-pp-(\d+)-cp-(\d+)-layer-(\d+)-router-debug\.npz$"
)
diagnostic_pattern = re.compile(
    r"^(trainer|h200)-rank-(\d+)-pp-(\d+)-cp-(\d+)-diagnostic-([a-z0-9_-]+)\.npz$"
)
diagnostic_manifest_name = "parity-diagnostics.json"
diagnostic_manifest_lock_name = "parity-diagnostics.lock"
root_identity_name = "parity-root-identity.json"
diagnostic_env = "PRIME_PARITY_DIAGNOSTICS"
diagnostic_tokens_env = "PRIME_PARITY_DIAGNOSTIC_TOKENS"
diagnostic_max_files_env = "PRIME_PARITY_DIAGNOSTIC_MAX_FILES"
diagnostic_max_bytes_env = "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES"
diagnostic_hidden_width = 6144
diagnostic_max_request_bytes = 16 * 1024 * 1024
parity_model_modules = (
    "prime_rl.trainer.model",
    "prime_rl.trainer.lora",
    "prime_rl.trainer.distributed.pipeline_parallel",
    "prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa",
    "prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention",
    "prime_rl.trainer.models.layers.moe",
    "prime_rl.trainer.models.layers.mxfp8_grouped_gemm",
    "prime_rl.trainer.models.layers.lora.multi_linear",
    "prime_rl.trainer.models.layers.lora.native_fp8_moe",
)
lifecycle_pattern = re.compile(r"^h200-lifecycle-rank-(\d+)\.json$")
chunk_pattern = re.compile(
    r"^(trainer|h200)-rank-(\d+)-pp-(\d+)-cp-(\d+)-layer-(\d+)-call-(\d+)\.npz$"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_tokens(token_ids: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(token_ids, dtype=np.int64))
    return sha256_bytes(memoryview(values).cast("B"))


def trace_session_identity(session: TraceSession) -> str:
    encoded = json.dumps(
        session.as_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sigmoid_router_reference(
    logits: np.ndarray,
    correction_bias: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32)
    correction_bias = np.asarray(correction_bias, dtype=np.float32)
    if logits.ndim != 2 or correction_bias.shape != (logits.shape[1],):
        raise ValueError("router logits or correction bias has the wrong shape")
    if topk < 1 or topk > logits.shape[1]:
        raise ValueError("topk is outside the expert count")
    scores = np.float32(0.5) * np.tanh(np.float32(0.5) * logits) + np.float32(0.5)
    choice = scores + correction_bias[None, :]
    ids = np.argsort(-choice, axis=1, kind="stable")[:, :topk]
    weights = np.take_along_axis(scores, ids, axis=1).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True, dtype=np.float32)
    return ids.astype(np.int32), weights.astype(np.float32)


def validate_sparse_routes(
    ids: np.ndarray,
    weights: np.ndarray,
    *,
    topk: int,
    experts: int,
    sum_tolerance: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids)
    weights = np.asarray(weights, dtype=np.float32)
    if ids.ndim != 2 or ids.shape[1] != topk or weights.shape != ids.shape:
        raise ValueError("router IDs and weights must have shape [tokens, topk]")
    if ids.dtype.kind not in "iu":
        raise ValueError("router IDs must use an integer dtype")
    if ids.size and (ids.min() < 0 or ids.max() >= experts):
        raise ValueError("router ID is outside the expert count")
    if ids.size and np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("one token selected the same expert more than once")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("router weights must be finite and positive")
    if weights.size:
        error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
        if error > sum_tolerance:
            raise ValueError(
                f"router weight sum error is {error:.6f}, limit is {sum_tolerance:.6f}"
            )
    return np.ascontiguousarray(ids.astype(np.int32)), np.ascontiguousarray(weights)

@dataclass(frozen=True)
class TraceSession:
    side: str
    fixture_sha256: str
    token_ids_sha256: str
    model_revision: str
    adapter_sha256: str
    image: str
    expected_tokens: int
    hidden_layers: int
    dense_layers: int
    topk: int
    experts: int
    route_output_scale: float
    weight_sum_tolerance: float

    def validate(self) -> None:
        if self.side not in {"trainer", "h200"}:
            raise ValueError("trace side must be trainer or h200")
        for name, value, length in (
            ("fixture checksum", self.fixture_sha256, 64),
            ("token checksum", self.token_ids_sha256, 64),
            ("model revision", self.model_revision, 40),
            ("adapter checksum", self.adapter_sha256, 64),
        ):
            if len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be {length} lowercase hex characters")
        if not self.image:
            raise ValueError("image must not be empty")
        if self.expected_tokens < 1 or self.hidden_layers < 1:
            raise ValueError("token and layer counts must be positive")
        if self.dense_layers < 0 or self.dense_layers >= self.hidden_layers:
            raise ValueError("dense layer count is invalid")
        if (
            not isinstance(self.route_output_scale, (int, float))
            or not np.isfinite(self.route_output_scale)
            or self.route_output_scale <= 0
        ):
            raise ValueError("route output scale must be finite and positive")
        if (
            self.topk < 1
            or self.experts < self.topk
            or not isinstance(self.weight_sum_tolerance, (int, float))
            or not np.isfinite(self.weight_sum_tolerance)
            or self.weight_sum_tolerance <= 0
        ):
            raise ValueError("router dimensions or scale are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": trace_schema_version, **self.__dict__}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TraceSession:
        if value.get("schema_version") != trace_schema_version:
            raise ValueError("router trace session schema does not match")
        result = cls(**{name: value[name] for name in cls.__dataclass_fields__})
        result.validate()
        return result


def trace_root() -> Path | None:
    raw = os.environ.get(trace_dir_env)
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{trace_dir_env} must be an absolute path")
    return path


def validated_trace_root() -> Path | None:
    root = trace_root()
    if root is None:
        return None
    configured = os.environ.get("PRIME_PARITY_QUALIFICATION_ROOT")
    if configured:
        return validate_isolated_parity_root(root, Path(configured))
    return root


def validate_isolated_parity_root(root: Path, configured: Path) -> Path:
    """Return an existing run root strictly below the qualification root."""
    if not root.is_absolute() or not configured.is_absolute():
        raise ValueError("parity roots must be absolute")
    def reject_symlink_components(path: Path) -> None:
        current = Path(path.anchor)
        for component in path.parts[1:]:
            current /= component
            try:
                mode = os.lstat(current).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise ValueError("parity roots cannot contain symlinks")

    reject_symlink_components(root)
    reject_symlink_components(configured)
    try:
        resolved_root = root.resolve(strict=True)
        resolved_configured = configured.resolve(strict=True)
    except OSError:
        raise
    if resolved_root == resolved_configured or resolved_configured not in resolved_root.parents:
        raise ValueError("parity trace root is outside the qualification root")
    return resolved_root


def _root_identity(root: Path) -> tuple[int, int]:
    stat = root.stat()
    if not root.is_dir():
        raise ValueError("parity trace root is not a directory")
    return stat.st_dev, stat.st_ino


def _validate_root_identity(root: Path) -> None:
    if (
        _bound_root_path == root
        and _bound_root_identity is not None
        and _bound_root_identity != _root_identity(root)
    ):
        raise ValueError("parity trace root was replaced")
    identity_path = root / root_identity_name
    if not identity_path.is_file():
        return
    data = json.loads(identity_path.read_text())
    if (data.get("device"), data.get("inode")) != _root_identity(root):
        raise ValueError("parity trace root was replaced")


def _write_root_identity(root: Path) -> None:
    device, inode = _root_identity(root)
    temporary = root / f".{root_identity_name}.{os.getpid()}"
    temporary.write_text(json.dumps({"device": device, "inode": inode}) + "\n")
    os.replace(temporary, root / root_identity_name)


def validate_diagnostic_content_length(
    content_length: str | None, transfer_encoding: str | None, limit: int
) -> int:
    if transfer_encoding and "chunked" in transfer_encoding.lower():
        raise ValueError("chunked diagnostic requests are not accepted")
    if content_length is None or not content_length.isdigit():
        raise ValueError("diagnostic request needs a valid Content-Length")
    size = int(content_length)
    if size < 1 or size > limit:
        raise ValueError("diagnostic request body is too large")
    return size


def diagnostics_enabled() -> bool:
    return os.environ.get(diagnostic_env) == "1"


def diagnostic_authorization_valid(
    authorization: str | None, expected_token: str | None
) -> bool:
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        return False
    if not isinstance(expected_token, str) or not expected_token:
        return False
    return secrets.compare_digest(authorization[7:], expected_token)


def consume_single_use_diagnostic_token(
    tokens: dict[str, str],
    expiry: dict[str, float],
    session_identity: str,
    authorization: str | None,
    now: float,
) -> bool:
    expected = tokens.get(session_identity)
    expires_at = expiry.get(session_identity, 0.0)
    if expires_at <= now or not diagnostic_authorization_valid(authorization, expected):
        if expires_at <= now:
            tokens.pop(session_identity, None)
            expiry.pop(session_identity, None)
        return False
    tokens.pop(session_identity, None)
    expiry.pop(session_identity, None)
    return True


def validate_h200_diagnostic_results(results: Any) -> dict[str, Any]:
    if not isinstance(results, list) or len(results) != 8 or not all(
        isinstance(item, dict) for item in results
    ):
        raise ValueError("H200 diagnostic did not return one result per TP8 worker")
    fields = ("session_identity", "request_identity", "layer", "input_shape")
    hashes = ("logits_sha256", "route_ids_sha256", "route_weights_sha256")
    first = results[0]
    for item in results[1:]:
        if any(item.get(field) != first.get(field) for field in fields + hashes):
            raise ValueError("H200 diagnostic worker results differ")
    if any(not isinstance(first.get(name), str) or len(first[name]) != 64 for name in hashes):
        raise ValueError("H200 diagnostic worker hashes are invalid")
    return first


def _diagnostic_limit(name: str) -> int:
    value = os.environ.get(name)
    if value is None or not value.isdigit() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer when diagnostics are enabled")
    return int(value)


def diagnostic_limits() -> dict[str, int]:
    if not diagnostics_enabled():
        return {}
    tokens = _diagnostic_limit(diagnostic_tokens_env)
    if tokens > 256:
        raise ValueError(f"{diagnostic_tokens_env} cannot exceed 256")
    return {
        "tokens": tokens,
        "max_files": _diagnostic_limit(diagnostic_max_files_env),
        "max_bytes": _diagnostic_limit(diagnostic_max_bytes_env),
    }


def _diagnostic_name(kind: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", kind):
        raise ValueError("diagnostic name is invalid")
    return kind


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mount_points() -> set[Path]:
    points: set[Path] = set()
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split(" - ", 1)[0].split()
        if len(fields) >= 5:
            points.add(Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[4])))
    return points


def _is_mounted(path: Path, mount_points: set[Path]) -> bool:
    return path in mount_points


def _trainer_host_name() -> str:
    host = os.environ.get("PRIME_TRAINER_HOST", "")
    if not re.fullmatch(r"n[0-9]+", host):
        raise ValueError("PRIME_TRAINER_HOST must identify the configured trainer host")
    return host


def _trainer_provenance(root: Path, owned_layers: set[int]) -> None:
    image_digest = os.environ.get("PRIME_TRAINER_IMAGE_DIGEST", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("PRIME_TRAINER_IMAGE_DIGEST must be an exact image digest")
    values = {
        name: os.environ.get(name, "")
        for name in (
            "PRIME_PARITY_PP_SIZE",
            "PRIME_PARITY_CP_SIZE",
            "PRIME_PARITY_BATCH_SIZE",
            "PRIME_PARITY_TRACE_MAX_CALLS",
            "PRIME_PARITY_TRACE_MAX_FILES",
            "PRIME_PARITY_TRACE_MAX_BYTES",
        )
    }
    if any(not value.isdigit() or int(value) < 1 for value in values.values()):
        raise ValueError("parity sizes, batch, and trace limits must be positive integers")
    if owned_layers != _expected_owned_layers:
        raise ValueError("observed router layers differ from trainer PP partition ownership")
    for name in parity_model_modules:
        importlib.import_module(name)
    mount_points = _mount_points()
    modules: dict[str, dict[str, str]] = {}
    for name in parity_model_modules:
        module = sys.modules.get(name)
        source = inspect.getsourcefile(module)
        if source is None:
            raise ValueError(f"required parity model module has no source: {name}")
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(f"required parity model module is not a file: {path}")
        location = "mounted" if path in mount_points else "unmounted"
        modules[name] = {"location": location, "sha256": _file_sha256(path)}
    payload = {
        "schema_version": trace_schema_version,
        "kind": "glm52_policy_parity_trainer_runtime",
        "image_digest": image_digest,
        "host": os.environ.get("PRIME_TRAINER_HOST", ""),
        "pp_rank": _process_pp_rank,
        "cp_rank": _process_cp_rank,
        "global_rank": _process_rank,
        "pp_size": int(values["PRIME_PARITY_PP_SIZE"]),
        "cp_size": int(values["PRIME_PARITY_CP_SIZE"]),
        "batch_size": int(values["PRIME_PARITY_BATCH_SIZE"]),
        "owned_layers": sorted(owned_layers),
        "trace_limits": {
            "max_calls": int(values["PRIME_PARITY_TRACE_MAX_CALLS"]),
            "max_files": int(values["PRIME_PARITY_TRACE_MAX_FILES"]),
            "max_bytes": int(values["PRIME_PARITY_TRACE_MAX_BYTES"]),
        },
        "modules": modules,
    }
    host = _trainer_host_name()
    filename = f"trainer-runtime-pp-{_process_pp_rank}-host-{host}.json"
    temporary = root / f".{filename}.{os.getpid()}"
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, root / filename)


def write_session(root: Path, session: TraceSession) -> None:
    session.validate()
    root.mkdir(parents=True, exist_ok=True)
    _validate_root_identity(root)
    path = root / session_name
    if path.exists():
        current = TraceSession.from_dict(json.loads(path.read_text()))
        if current != session:
            raise ValueError("router trace session does not match the existing session")
        _write_root_identity(root)
        return
    temporary = root / f".{session_name}.{os.getpid()}"
    temporary.write_text(json.dumps(session.as_dict(), sort_keys=True) + "\n")
    os.replace(temporary, path)
    _write_root_identity(root)


def read_session(root: Path) -> TraceSession:
    _validate_root_identity(root)
    return TraceSession.from_dict(json.loads((root / session_name).read_text()))


def read_diagnostic_manifest(root: Path, session: TraceSession) -> dict[str, Any]:
    _validate_root_identity(root)
    path = root / diagnostic_manifest_name
    if not path.is_file():
        raise ValueError("parity diagnostic session manifest is missing")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("session_identity") != trace_session_identity(session)
        or manifest.get("side") != session.side
        or not isinstance(manifest.get("entries"), list)
    ):
        raise ValueError("parity diagnostic session manifest differs")
    return manifest


def clear_trace(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _validate_root_identity(root)
    for path in root.iterdir():
        if (
            path.name in {
                session_name,
                runtime_marker_name,
                trainer_provenance_name,
                root_identity_name,
            }
            or path.name.startswith("trainer-runtime-pp-")
            or chunk_pattern.match(path.name)
            or router_debug_pattern.match(path.name)
            or diagnostic_pattern.match(path.name)
            or path.name == diagnostic_manifest_name
            or path.name == diagnostic_manifest_lock_name
            or lifecycle_pattern.match(path.name)
        ):
            if not path.is_file():
                raise ValueError(f"router trace path is not a file: {path}")
            path.unlink()


def configure_trainer_trace(
    config: Any, parallel_dims: Any, world: Any, model: Any = None
) -> bool:
    root = validated_trace_root()
    if root is None:
        return False
    configured = os.environ.get("PRIME_PARITY_QUALIFICATION_ROOT")
    if configured:
        validate_isolated_parity_root(root, Path(configured))
    expected_root = os.environ.get("PRIME_PARITY_TRACE_ROOT")
    if expected_root is None or root != Path(expected_root) or not re.fullmatch(
        r"/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces/[A-Za-z0-9_-]+",
        str(root),
    ):
        raise ValueError("parity trace root is not the configured per-run directory")
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise ValueError(f"trainer parity trace directory is missing or not writable: {root}")
    if not (root / session_name).is_file():
        raise ValueError(f"trainer parity trace session is missing: {root / session_name}")
    if not config.enable_token_export:
        raise ValueError("trainer router trace requires enable_token_export=true")
    session = read_session(root)
    if session.side != "trainer":
        raise ValueError("trainer router trace session side must be trainer")
    if model is None:
        raise ValueError("trainer parity trace requires the initialized model")
    from prime_rl.trainer.model import get_language_model
    from prime_rl.trainer.distributed.pipeline_parallel import stage_layer_range

    language_model = get_language_model(model)
    model_config = getattr(language_model, "config", None) or getattr(model, "config", None)
    pp_rank = (
        parallel_dims.world_mesh["pp"].get_local_rank()
        if parallel_dims.pp_enabled
        else 0
    )
    layer_start, layer_end = stage_layer_range(
        model_config.num_hidden_layers,
        pp_rank,
        parallel_dims.pp,
        config=model_config,
    )
    expected_owned_layers = set(range(max(layer_start, session.dense_layers), layer_end))
    cp_rank = (
        parallel_dims.world_mesh["cp"].get_local_rank()
        if parallel_dims.cp_enabled
        else 0
    )
    configured = configure_process_trace(
        "trainer",
        rank=world.rank,
        pp_rank=pp_rank,
        cp_rank=cp_rank,
        world_size=world.world_size,
        pp_size=int(parallel_dims.pp),
        cp_size=int(parallel_dims.cp),
    )
    if configured:
        if os.environ.get("PRIME_PARITY_PP_SIZE") != str(parallel_dims.pp):
            raise ValueError("configured parity PP size differs from trainer mesh")
        if os.environ.get("PRIME_PARITY_CP_SIZE") != str(parallel_dims.cp):
            raise ValueError("configured parity CP size differs from trainer mesh")
        global _expected_owned_layers
        _expected_owned_layers = expected_owned_layers
        if diagnostics_enabled():
            diagnostic_limits()
            configure_hidden_diagnostics(language_model)
    return configured


_process_side: str | None = None
_process_rank = 0
_process_pp_rank = 0
_process_cp_rank = 0
_process_world_size = 1
_process_pp_size = 1
_process_cp_size = 1
_process_session: TraceSession | None = None
_process_request: str | None = None
_bound_root_identity: tuple[int, int] | None = None
_bound_root_path: Path | None = None
_active_h200_session: str | None = None
_runtime_calls: dict[int, int] = {}
_runtime_id_sources: dict[int, str] = {}
_runtime_scales: dict[int, float] = {}
_calls: dict[int, int] = {}
_written_files = 0
_written_bytes = 0
_debug_written = False
_debug_written_bytes = 0
_debug_written_session: str | None = None
_owned_layers: set[int] = set()
_provenance_written = False
_expected_owned_layers: set[int] = set()
_diagnostic_files = 0
_diagnostic_bytes = 0
_diagnostic_session: str | None = None
_h200_diagnostic_active = False
_diagnostic_hooks: list[Any] = []
_write_lock = threading.Lock()


def configure_process_trace(
    side: str,
    *,
    rank: int,
    pp_rank: int = 0,
    cp_rank: int = 0,
    world_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
) -> bool:
    global _process_side, _process_rank, _process_pp_rank, _process_cp_rank
    global _process_world_size, _process_pp_size, _process_cp_size
    global _process_session, _process_request, _active_h200_session
    global _bound_root_identity, _bound_root_path
    root = validated_trace_root()
    if root is None:
        return False
    if side not in {"trainer", "h200"}:
        raise ValueError("trace side must be trainer or h200")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("trace process rank or world size is invalid")
    if side == "trainer":
        if pp_size < 1 or cp_size < 1 or world_size != pp_size * cp_size:
            raise ValueError("parity PP/CP sizes do not match the trainer world")
        if rank != pp_rank * cp_size + cp_rank:
            raise ValueError("trainer global rank does not match PP/CP rank")
    _process_rank = rank
    _process_pp_rank = pp_rank
    _process_cp_rank = cp_rank
    _process_world_size = world_size
    _process_pp_size = pp_size
    _process_cp_size = cp_size
    _process_side = side
    _process_session = None
    _process_request = None
    _active_h200_session = None
    _bound_root_identity = _root_identity(root) if root.is_dir() else None
    _bound_root_path = root
    _calls.clear()
    global _written_files, _written_bytes, _debug_written, _debug_written_bytes
    global _debug_written_session
    global _owned_layers, _provenance_written
    global _diagnostic_files, _diagnostic_bytes, _diagnostic_session
    global _h200_diagnostic_active
    global _diagnostic_hooks
    _written_files = 0
    _written_bytes = 0
    _debug_written = False
    _debug_written_bytes = 0
    _debug_written_session = None
    _owned_layers = set()
    _provenance_written = False
    _diagnostic_files = 0
    _diagnostic_bytes = 0
    _diagnostic_session = None
    _h200_diagnostic_active = False
    for handle in _diagnostic_hooks:
        handle.remove()
    _diagnostic_hooks = []
    _runtime_calls.clear()
    _runtime_id_sources.clear()
    _runtime_scales.clear()
    return True


def _diagnostic_array(value: Any, limit: int) -> tuple[np.ndarray, str]:
    tensor = value.detach()
    source_dtype = str(getattr(tensor, "dtype", "unknown"))
    if getattr(tensor, "ndim", 0) < 2:
        raise ValueError("diagnostic tensor must have at least two dimensions")
    tensor = tensor.reshape(-1, tensor.shape[-1])[:limit]
    tensor = tensor.to(dtype=__import__("torch").float32, device="cpu")
    array = np.ascontiguousarray(tensor.numpy())
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("diagnostic tensor is invalid")
    return array, source_dtype


def _diagnostic_tensor_metadata(
    value: Any, limit: int, expected_width: int | None
) -> tuple[int, str, int]:
    tensor = value.detach()
    shape = tuple(int(item) for item in getattr(tensor, "shape", ()))
    if len(shape) < 2 or (expected_width is not None and shape[-1] != expected_width):
        raise ValueError("diagnostic tensor has an invalid width")
    token_count = 1
    for size in shape[:-1]:
        token_count *= size
    token_count = min(token_count, limit)
    if token_count < 1:
        raise ValueError("diagnostic tensor has no tokens")
    source_dtype = str(getattr(tensor, "dtype", "unknown"))
    estimated_bytes = token_count * shape[-1] * 4 + 1024 * 1024
    return token_count, source_dtype, estimated_bytes


def write_hidden_diagnostic(
    kind: str, value: Any, *, expected_width: int | None = diagnostic_hidden_width
) -> None:
    """Write one bounded activation capture when explicitly enabled."""
    global _diagnostic_files, _diagnostic_bytes, _diagnostic_session
    root = validated_trace_root()
    if root is None or _process_side is None or not diagnostics_enabled():
        return
    if _process_side == "h200" and _active_h200_session is None:
        return
    if _process_side == "h200" and _process_rank != 0:
        return
    kind = _diagnostic_name(kind)
    entry_name = (
        f"{kind}-rank-{_process_rank}-pp-{_process_pp_rank}-cp-{_process_cp_rank}"
    )
    session = read_session(root)
    limits = diagnostic_limits()
    with _write_lock:
        session_identity = trace_session_identity(session)
        if _diagnostic_session != session_identity:
            _diagnostic_files = 0
            _diagnostic_bytes = 0
            _diagnostic_session = session_identity
        token_count, source_dtype, estimated_bytes = _diagnostic_tensor_metadata(
            value, limits["tokens"], expected_width
        )
        filename = (
            f"{_process_side}-rank-{_process_rank}-pp-{_process_pp_rank}-"
            f"cp-{_process_cp_rank}-diagnostic-{kind}.npz"
        )
        path = root / filename
        manifest_path = root / diagnostic_manifest_name
        with (root / diagnostic_manifest_lock_name).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("session_identity") != session_identity:
                    raise ValueError("diagnostic manifest session differs")
                entries = manifest.get("entries", [])
                if not isinstance(entries, list):
                    raise ValueError("diagnostic manifest entries are invalid")
            else:
                entries = []
                manifest = {
                    "schema_version": 1,
                    "kind": "glm52_policy_parity_diagnostics",
                    "session_identity": session_identity,
                    "side": _process_side,
                    "entries": entries,
                }
            if any(entry.get("name") == entry_name for entry in entries):
                return
            if path.exists():
                raise FileExistsError(f"diagnostic output already exists: {path}")
            existing_bytes = sum(int(entry.get("bytes", 0)) for entry in entries)
            if (
                len(entries) + 1 > limits["max_files"]
                or existing_bytes + estimated_bytes > limits["max_bytes"]
            ):
                raise ValueError("parity diagnostic file or byte limit exceeded")
            array, source_dtype = _diagnostic_array(value, limits["tokens"])
            if array.shape[0] != token_count:
                raise ValueError("diagnostic tensor changed during capture")
            temporary = root / f".{filename}.{os.getpid()}"
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    values=array,
                    session_identity=np.asarray(session_identity, dtype="<U64"),
                    name=np.asarray(entry_name, dtype="<U128"),
                    source_dtype=np.asarray(source_dtype, dtype="<U32"),
                    fixture_sha256=np.asarray(session.fixture_sha256, dtype="<U64"),
                    token_ids_sha256=np.asarray(session.token_ids_sha256, dtype="<U64"),
                    model_revision=np.asarray(session.model_revision, dtype="<U40"),
                    adapter_sha256=np.asarray(session.adapter_sha256, dtype="<U64"),
                )
            size = temporary.stat().st_size
            if size > limits["max_bytes"] or size > estimated_bytes:
                temporary.unlink()
                raise ValueError("parity diagnostic file or byte limit exceeded")
            os.replace(temporary, path)
            entry = {
                "name": entry_name,
                "file": filename,
                "shape": list(array.shape),
                "source_dtype": source_dtype,
                "sha256": sha256_bytes(memoryview(array).cast("B")),
                "bytes": size,
            }
            entries.append(entry)
            manifest["entries"] = entries
            manifest_temporary = root / f".{diagnostic_manifest_name}.{os.getpid()}"
            manifest_temporary.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            os.replace(manifest_temporary, manifest_path)
        _diagnostic_files += 1
        _diagnostic_bytes += size


def h200_diagnostic_is_active() -> bool:
    return _h200_diagnostic_active


def _h200_router_for_layer(model: Any, layer: int) -> tuple[Any, Any]:
    modules = model.modules() if hasattr(model, "modules") else ()
    for module in modules:
        router = getattr(module, "router", None)
        if router is not None and getattr(router, "_prime_parity_layer", None) == layer:
            return module, router
    raise ValueError(f"H200 router layer {layer} is not available")


def _router_logits(owner: Any, router: Any, hidden: Any) -> Any:
    expected_bias = getattr(router, "e_score_correction_bias", None)
    for candidate in (getattr(owner, "gate", None), getattr(router, "gate", None)):
        if candidate is None:
            continue
        operation = candidate
        if not callable(operation):
            weight = getattr(operation, "weight", None)
            if weight is None:
                continue
            import torch.nn.functional as functional

            def operation(values: Any, weight: Any = weight) -> Any:
                return functional.linear(values, weight)
        result = operation(hidden)
        output_bias = None
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("H200 gate returned an invalid result tuple")
            result, output_bias = result
        if output_bias is not None:
            if expected_bias is None or not __import__("torch").equal(
                output_bias.to(dtype=__import__("torch").float32),
                expected_bias.to(dtype=__import__("torch").float32),
            ):
                raise ValueError("H200 gate bias differs from router correction bias")
        elif expected_bias is not None and getattr(candidate, "bias", None) is not None:
            raise ValueError("H200 gate omitted its configured correction bias")
        return result
    for candidate in (getattr(owner, "layer", None),):
        if candidate is None:
            continue
        operation = getattr(candidate, "gate", None)
        if callable(operation):
            result = operation(hidden)
            if isinstance(result, tuple):
                result = result[0]
            return result
    raise ValueError("H200 router does not expose its gate operation")


def run_h200_router_diagnostic(model_runner: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one bounded captured-input router check inside the H200 worker."""
    if not diagnostics_enabled():
        raise ValueError("H200 parity diagnostics are disabled")
    if _process_side != "h200":
        raise ValueError("H200 router diagnostics require an H200 process")
    root = validated_trace_root()
    if root is None:
        raise ValueError("H200 router diagnostics have no trace directory")
    session = read_session(root)
    session_identity = payload.get("session_identity")
    source_session_identity = payload.get("source_session_identity")
    request_identity = payload.get("request_identity")
    if session.side != "h200" or trace_session_identity(session) != session_identity:
        raise ValueError("H200 diagnostic session differs")
    if not isinstance(source_session_identity, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_session_identity
    ):
        raise ValueError("trainer diagnostic session identity is invalid")
    for name in ("fixture_sha256", "token_ids_sha256", "adapter_sha256"):
        if payload.get(f"source_{name}") != getattr(session, name):
            raise ValueError(f"H200 diagnostic {name} differs")
    if payload.get("source_model_revision") != session.model_revision:
        raise ValueError("H200 diagnostic model revision differs")
    if not isinstance(request_identity, str) or not re.fullmatch(r"[0-9a-f]{32}", request_identity):
        raise ValueError("H200 diagnostic request identity is invalid")
    layer = payload.get("layer")
    if not isinstance(layer, int) or layer != session.dense_layers:
        raise ValueError("H200 diagnostic must target the first routed layer")
    encoded = payload.get("router_input")
    if not isinstance(encoded, dict):
        raise ValueError("H200 diagnostic input is missing")
    expected_bytes, _ = validate_encoded_diagnostic_input(encoded)
    input_array = decoded_array(encoded)
    if input_array.nbytes != expected_bytes:
        raise ValueError("H200 diagnostic input byte count differs")
    expected_input_hash = payload.get("router_input_sha256")
    actual_input_hash = sha256_bytes(
        memoryview(np.ascontiguousarray(input_array)).cast("B")
    )
    if expected_input_hash != actual_input_hash:
        raise ValueError("H200 diagnostic input checksum differs")
    source_dtype = payload.get("source_dtype")
    if source_dtype not in {"torch.bfloat16", "bfloat16"}:
        raise ValueError("H200 diagnostic input is not a BF16 trainer capture")
    limits = diagnostic_limits()
    if input_array.ndim != 2 or input_array.shape[0] > limits["tokens"]:
        raise ValueError("H200 diagnostic input exceeds the configured token limit")
    if input_array.shape[1] < 1 or not np.isfinite(input_array).all():
        raise ValueError("H200 diagnostic input is invalid")
    global _h200_diagnostic_active
    with h200_diagnostic_request(session_identity, request_identity):
        torch = __import__("torch")
        model = getattr(model_runner, "model", model_runner)
        model = getattr(model, "runnable", model)
        owner, router = _h200_router_for_layer(model, layer)
        parameter = next(iter(model.parameters()), None)
        device = getattr(parameter, "device", "cuda")
        hidden = torch.as_tensor(input_array, dtype=torch.bfloat16, device=device)
        _h200_diagnostic_active = True
        try:
            logits = _router_logits(owner, router, hidden)
            route_weights, route_ids = router.select_experts(hidden, logits)
        finally:
            _h200_diagnostic_active = False
        logits = logits.to(torch.float32)
        scores = 0.5 * torch.tanh(0.5 * logits) + 0.5
        correction_bias = getattr(router, "e_score_correction_bias", None)
        selection_scores = scores
        if correction_bias is not None:
            selection_scores = selection_scores + correction_bias.to(torch.float32)
        write_hidden_diagnostic(f"layer{layer}_h200_input", hidden)
        write_hidden_diagnostic(
            f"layer{layer}_h200_logits", logits, expected_width=None
        )
        write_hidden_diagnostic(
            f"layer{layer}_h200_scores", scores, expected_width=None
        )
        write_hidden_diagnostic(
            f"layer{layer}_h200_selection_scores", selection_scores, expected_width=None
        )
        write_hidden_diagnostic(
            f"layer{layer}_h200_route_ids", route_ids, expected_width=None
        )
        write_hidden_diagnostic(
            f"layer{layer}_h200_route_weights", route_weights, expected_width=None
        )
        def tensor_hash(value: Any, dtype: Any) -> str:
            array = value.detach().to(dtype=dtype, device="cpu").contiguous().numpy()
            return sha256_bytes(memoryview(np.ascontiguousarray(array)).cast("B"))

        return {
            "session_identity": session_identity,
            "source_session_identity": source_session_identity,
            "request_identity": request_identity,
            "layer": layer,
            "input_shape": list(input_array.shape),
            "logits_sha256": tensor_hash(logits, torch.float32),
            "route_ids_sha256": tensor_hash(route_ids, torch.int32),
            "route_weights_sha256": tensor_hash(route_weights, torch.float32),
        }


def _hidden_stream(output: Any, name: str) -> Any:
    if hasattr(output, "detach"):
        return output
    if (
        isinstance(output, tuple)
        and len(output) == 2
        and hasattr(output[0], "detach")
        and (output[1] is None or hasattr(output[1], "detach"))
    ):
        return output[0]
    raise ValueError(f"{name} output is not a known hidden-state stream")


def _slice_diagnostic_tokens(value: Any, limit: int = 256) -> Any:
    shape = tuple(int(size) for size in getattr(value, "shape", ()))
    if len(shape) < 2:
        raise ValueError("diagnostic boundary value is not a token tensor")
    return value[:limit]


def _capture_embedding(_module: Any, _inputs: Any, output: Any) -> None:
    hidden = _hidden_stream(output, "embedding")
    write_hidden_diagnostic("embedding", hidden)
    write_hidden_diagnostic("layer0_input", hidden)


def _capture_attention(_module: Any, _inputs: Any, output: Any, layer: int) -> None:
    write_hidden_diagnostic(
        f"layer{layer}_attention_raw", _hidden_stream(output, "attention")
    )


def _capture_mlp(_module: Any, _inputs: Any, output: Any, layer: int) -> None:
    if not hasattr(output, "detach"):
        raise ValueError("MLP output is not a tensor")
    write_hidden_diagnostic(f"layer{layer}_mlp_raw", output)


def _capture_decoder_layer(_module: Any, _inputs: Any, output: Any, layer: int) -> None:
    hidden = _hidden_stream(output, "decoder layer")
    write_hidden_diagnostic(f"layer{layer}_output", hidden)


def _capture_trainer_decoder_input(_module: Any, inputs: Any, layer: int) -> None:
    if not isinstance(inputs, tuple) or not inputs or not hasattr(inputs[0], "detach"):
        raise ValueError("trainer decoder input is not a hidden-state tensor")
    write_hidden_diagnostic(f"layer{layer}_input", inputs[0])


def configure_hidden_diagnostics(language_model: Any) -> None:
    """Install bounded hooks for the first dense layers of the local PP stage."""
    if not diagnostics_enabled():
        return
    root_model = language_model
    if not hasattr(root_model, "embed_tokens") and hasattr(root_model, "model"):
        root_model = root_model.model
    layers = getattr(root_model, "layers", None)
    if layers is None:
        raise ValueError("parity diagnostics cannot find model layers")
    embed = getattr(root_model, "embed_tokens", None)
    if embed is not None:
        _diagnostic_hooks.append(
            embed.register_forward_hook(_capture_embedding)
        )
    items = layers.items() if hasattr(layers, "items") else enumerate(layers)
    for raw_layer, layer in items:
        layer_number = int(raw_layer)
        if layer_number > 2:
            continue
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            _diagnostic_hooks.append(
                attention.register_forward_hook(
                    lambda module, inputs, output, number=layer_number: _capture_attention(
                        module, inputs, output, number
                    )
                )
            )
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            _diagnostic_hooks.append(
                mlp.register_forward_hook(
                    lambda module, inputs, output, number=layer_number: _capture_mlp(
                        module, inputs, output, number
                    )
                )
            )
        _diagnostic_hooks.append(
            layer.register_forward_pre_hook(
                lambda module, inputs, number=layer_number: _capture_trainer_decoder_input(
                    module, inputs, number
                )
            )
        )
        _diagnostic_hooks.append(
            layer.register_forward_hook(
                lambda module, inputs, output, number=layer_number: _capture_decoder_layer(
                    module, inputs, output, number
                )
            )
        )


def _capture_h200_attention_boundary(
    _module: Any, _inputs: Any, output: Any, layer: int
) -> None:
    if not hasattr(output, "detach"):
        raise ValueError("H200 attention output is not a tensor")
    write_hidden_diagnostic(f"layer{layer}_attention_raw", output)


def _capture_h200_decoder_input(
    _module: Any, inputs: Any, layer: int
) -> None:
    if not isinstance(inputs, tuple) or len(inputs) < 3:
        raise ValueError("H200 decoder input does not contain hidden and residual")
    hidden, residual = inputs[1], inputs[2]
    if not hasattr(hidden, "detach"):
        raise ValueError("H200 decoder input hidden state is not a tensor")
    hidden = _slice_diagnostic_tokens(hidden)
    if residual is not None:
        if not hasattr(residual, "detach"):
            raise ValueError("H200 decoder input residual is not a tensor")
        residual = _slice_diagnostic_tokens(residual)
        hidden = hidden + residual
    write_hidden_diagnostic(f"layer{layer}_input", hidden)


def _capture_h200_mlp_boundary(
    _module: Any, _inputs: Any, output: Any, layer: int
) -> None:
    if not hasattr(output, "detach"):
        raise ValueError("H200 dense MLP output is not a tensor")
    write_hidden_diagnostic(f"layer{layer}_mlp_raw", output)


def _capture_h200_layer_boundary(
    _module: Any, _inputs: Any, output: Any, layer: int
) -> None:
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not hasattr(output[0], "detach")
        or not hasattr(output[1], "detach")
    ):
        raise ValueError("H200 decoder boundary is not a hidden/residual pair")
    hidden = _slice_diagnostic_tokens(output[0])
    residual = _slice_diagnostic_tokens(output[1])
    write_hidden_diagnostic(f"layer{layer}_output", hidden + residual)


def configure_h200_boundary_diagnostics(model: Any) -> None:
    if not diagnostics_enabled():
        return
    root_model = getattr(model, "model", model)
    embed = getattr(root_model, "embed_tokens", None)
    layers = getattr(root_model, "layers", None)
    if embed is None or layers is None or not hasattr(embed, "register_forward_hook"):
        raise ValueError("H200 diagnostics cannot find the model embedding and layers")
    _diagnostic_hooks.append(embed.register_forward_hook(_capture_embedding))
    layer_items = layers.items() if hasattr(layers, "items") else enumerate(layers)
    selected: dict[int, Any] = {}
    for raw_layer, layer in layer_items:
        layer_number = int(getattr(layer, "layer_idx", raw_layer))
        if layer_number in {0, 1, 2}:
            selected[layer_number] = layer
    if set(selected) != {0, 1, 2}:
        raise ValueError("H200 diagnostics require local logical layers 0, 1, and 2")
    for layer_number in range(3):
        layer = selected[layer_number]
        if not hasattr(layer, "register_forward_pre_hook"):
            raise ValueError("H200 diagnostics cannot find decoder input boundary")
        _diagnostic_hooks.append(
            layer.register_forward_pre_hook(
                lambda module, inputs, number=layer_number: _capture_h200_decoder_input(
                    module, inputs, number
                )
            )
        )
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if attention is None or not hasattr(attention, "register_forward_hook"):
            raise ValueError("H200 diagnostics cannot find self-attention boundary")
        if mlp is None or not hasattr(mlp, "register_forward_hook"):
            raise ValueError("H200 diagnostics cannot find dense MLP boundary")
        if getattr(mlp, "router", None) is not None:
            raise ValueError("H200 diagnostics requires a dense MLP for layers 0-2")
        _diagnostic_hooks.append(
            attention.register_forward_hook(
                lambda module, inputs, output, number=layer_number: _capture_h200_attention_boundary(
                    module, inputs, output, number
                )
            )
        )
        _diagnostic_hooks.append(
            mlp.register_forward_hook(
                lambda module, inputs, output, number=layer_number: _capture_h200_mlp_boundary(
                    module, inputs, output, number
                )
            )
        )
        _diagnostic_hooks.append(
            layer.register_forward_hook(
                lambda module, inputs, output, number=layer_number: _capture_h200_layer_boundary(
                    module, inputs, output, number
                )
            )
        )


def _lifecycle_path(root: Path, rank: int) -> Path:
    return root / f"h200-lifecycle-rank-{rank}.json"


def _write_lifecycle(root: Path, marker: dict[str, Any]) -> None:
    path = _lifecycle_path(root, int(marker["rank"]))
    temporary = root / f".{path.name}.{os.getpid()}"
    temporary.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, path)


def mark_h200_trace_flags(session_identity: str, request_identity: str) -> None:
    root = validated_trace_root()
    if root is None:
        raise ValueError("H200 router trace request has no trace directory")
    session = read_session(root)
    if trace_session_identity(session) != session_identity:
        raise ValueError("H200 router trace request session does not match")
    if not isinstance(request_identity, str) or len(request_identity) != 32:
        raise ValueError("H200 router trace request identity is invalid")
    _write_lifecycle(
        root,
        {
            "session_identity": session_identity,
            "request_identity": request_identity,
            "rank": _process_rank,
            "world_size": _process_world_size,
            "flags_received": True,
            "context_entered": False,
            "router_calls": 0,
        },
    )


def _enter_h200_trace_context(
    session_identity: str, request_identity: str
) -> None:
    root = validated_trace_root()
    if root is None:
        raise ValueError("H200 router trace request has no trace directory")
    path = _lifecycle_path(root, _process_rank)
    if not path.is_file():
        raise ValueError("H200 router trace request flags were not received")
    marker = json.loads(path.read_text())
    if (
        marker.get("session_identity") != session_identity
        or marker.get("request_identity") != request_identity
        or marker.get("rank") != _process_rank
        or marker.get("world_size") != _process_world_size
        or marker.get("flags_received") is not True
    ):
        raise ValueError("H200 router trace request marker differs")
    marker["context_entered"] = True
    _write_lifecycle(root, marker)


@contextmanager
def h200_trace_request(session_identity: str, request_identity: str):
    global _process_session, _process_request, _active_h200_session
    _enter_h200_trace_context(session_identity, request_identity)
    if _process_side is None:
        yield
        return
    if _process_side != "h200":
        raise ValueError("H200 router trace request used a non-H200 process")
    root = validated_trace_root()
    if root is None:
        raise ValueError("H200 router trace request has no trace directory")
    session = read_session(root)
    if trace_session_identity(session) != session_identity:
        raise ValueError("H200 router trace request session does not match")
    with _write_lock:
        if _active_h200_session is not None:
            raise ValueError("H200 router trace request is already active")
        if _process_request != request_identity:
            _calls.clear()
            _runtime_calls.clear()
            _runtime_id_sources.clear()
            _runtime_scales.clear()
            _process_request = request_identity
        _process_session = session
        _active_h200_session = session_identity
    try:
        yield
    finally:
        with _write_lock:
            _active_h200_session = None


@contextmanager
def h200_diagnostic_request(session_identity: str, request_identity: str):
    global _process_session, _process_request, _active_h200_session
    root = validated_trace_root()
    if root is None or _process_side != "h200":
        raise ValueError("H200 diagnostic requires an H200 trace process")
    session = read_session(root)
    if session.side != "h200" or trace_session_identity(session) != session_identity:
        raise ValueError("H200 diagnostic session does not match the session manifest")
    with _write_lock:
        if _active_h200_session is not None:
            raise ValueError("H200 diagnostic session is already active")
        _process_session = session
        _process_request = request_identity
        _active_h200_session = session_identity
    try:
        yield session
    finally:
        with _write_lock:
            _active_h200_session = None


def h200_trace_is_active() -> bool:
    return _process_side == "h200" and _active_h200_session is not None


def read_h200_runtime_marker(
    root: Path, session: TraceSession
) -> dict[str, Any]:
    lifecycle_paths = sorted(
        path for path in root.iterdir() if lifecycle_pattern.match(path.name)
    )
    if not lifecycle_paths:
        raise ValueError("H200 router trace request flags not received")
    lifecycle = [json.loads(path.read_text()) for path in lifecycle_paths]
    world_sizes = {int(marker.get("world_size", 0)) for marker in lifecycle}
    if len(world_sizes) != 1:
        raise ValueError("H200 router trace rank coverage differs")
    world_size = world_sizes.pop()
    ranks = {int(marker.get("rank", -1)) for marker in lifecycle}
    if ranks != set(range(world_size)):
        raise ValueError("H200 router trace rank coverage is incomplete")
    for marker in lifecycle:
        if marker.get("session_identity") != trace_session_identity(session):
            raise ValueError("H200 router trace lifecycle session differs")
        if marker.get("flags_received") is not True:
            raise ValueError("H200 router trace request flags not received")
        if marker.get("context_entered") is not True:
            raise ValueError("H200 router trace context not entered")
    rank_zero = next((marker for marker in lifecycle if marker.get("rank") == 0), None)
    if rank_zero is None:
        raise ValueError("H200 router trace request flags not received on rank 0")
    if int(rank_zero.get("router_calls", 0)) < 1:
        raise ValueError("H200 router trace router not called")
    marker = json.loads((root / runtime_marker_name).read_text())
    if marker.get("session_identity") != trace_session_identity(session):
        raise ValueError("H200 router runtime marker session differs")
    request_identity = marker.get("request_identity")
    if not isinstance(request_identity, str) or len(request_identity) != 32:
        raise ValueError("H200 router runtime marker request is invalid")
    raw_calls = marker.get("layer_calls")
    if not isinstance(raw_calls, dict):
        raise ValueError("H200 router runtime marker calls are missing")
    calls = {int(layer): int(count) for layer, count in raw_calls.items()}
    expected_layers = set(range(session.dense_layers, session.hidden_layers))
    if set(calls) != expected_layers or any(count < 1 for count in calls.values()):
        raise ValueError("H200 router runtime marker does not cover every routed layer")
    raw_sources = marker.get("id_sources")
    if not isinstance(raw_sources, dict):
        raise ValueError("H200 router runtime marker ID sources are missing")
    sources = {int(layer): str(source) for layer, source in raw_sources.items()}
    if set(sources) != expected_layers:
        raise ValueError("H200 router runtime marker ID sources are incomplete")
    if any(source != "modular_actual" for source in sources.values()):
        raise ValueError("router IDs are not from the executed MoE path")
    raw_scales = marker.get("route_output_scales")
    if not isinstance(raw_scales, dict):
        raise ValueError("H200 route output scales are missing")
    scales = {int(layer): float(scale) for layer, scale in raw_scales.items()}
    if set(scales) != expected_layers:
        raise ValueError("H200 route output scales are incomplete")
    if any(not np.isfinite(scale) or scale <= 0 for scale in scales.values()):
        raise ValueError("H200 route output scale is invalid")
    if any(scale != session.route_output_scale for scale in scales.values()):
        raise ValueError("H200 route output scale differs from session")
    return {
        "session_identity": marker["session_identity"],
        "request_identity": request_identity,
        "layer_calls": {str(layer): calls[layer] for layer in sorted(calls)},
        "id_sources": {str(layer): sources[layer] for layer in sorted(sources)},
        "route_output_scales": {
            str(layer): scales[layer] for layer in sorted(scales)
        },
        "ranks": sorted(ranks),
    }


def write_route_chunk(
    layer: int,
    ids: Any,
    weights: Any,
    *,
    route_output_scale: float,
    id_source: str = "modular_actual",
) -> None:
    global _process_session
    if id_source != "modular_actual":
        raise ValueError("router trace ID source is invalid")
    root = validated_trace_root()
    if root is None or _process_side is None:
        return
    if _process_side == "h200" and _active_h200_session is None:
        return
    if _process_side == "h200" and _process_rank != 0:
        return
    torch = __import__("torch")
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and compiler.is_compiling():
        return
    cuda = getattr(torch, "cuda", None)
    if (
        cuda is not None
        and cuda.is_available()
        and cuda.is_current_stream_capturing()
    ):
        return
    ids_array = ids.detach().to(dtype=torch.int32, device="cpu").numpy()
    weight_array = weights.detach().to(dtype=torch.float32, device="cpu").numpy()
    session = read_session(root)
    if session.side != _process_side:
        raise ValueError("router trace process side differs from session")
    if _process_side == "h200":
        if _process_session != session:
            raise ValueError("H200 router trace session changed during the request")
        if trace_session_identity(session) != _active_h200_session:
            raise ValueError("H200 router trace active session differs from disk")
    ids_array, weight_array = validate_sparse_routes(
        ids_array,
        weight_array,
        topk=session.topk,
        experts=session.experts,
        sum_tolerance=session.weight_sum_tolerance,
    )
    if (
        not isinstance(route_output_scale, (int, float))
        or not np.isfinite(route_output_scale)
        or route_output_scale <= 0
    ):
        raise ValueError("route output scale must be finite and positive")
    if float(route_output_scale) != session.route_output_scale:
        raise ValueError("route output scale differs from the trace session")
    global _written_files, _written_bytes, _owned_layers, _provenance_written
    estimate = int(ids_array.nbytes + weight_array.nbytes + 4096)
    if _process_side == "trainer":
        limits = {
            name: int(os.environ.get(name, "0"))
            for name in (
                "PRIME_PARITY_TRACE_MAX_CALLS",
                "PRIME_PARITY_TRACE_MAX_FILES",
                "PRIME_PARITY_TRACE_MAX_BYTES",
            )
        }
        if min(limits.values()) < 1:
            raise ValueError("parity trace limits are missing")
        if sum(_calls.values()) >= limits["PRIME_PARITY_TRACE_MAX_CALLS"]:
            raise ValueError("parity router trace call limit exceeded")
        if (
            _written_files >= limits["PRIME_PARITY_TRACE_MAX_FILES"]
            or _written_bytes + estimate > limits["PRIME_PARITY_TRACE_MAX_BYTES"]
        ):
            raise ValueError("parity router trace file or byte limit exceeded")
    _owned_layers.add(layer)
    with _write_lock:
        if _process_side != "h200" and _process_session != session:
            _calls.clear()
            _process_session = session
        call = _calls.get(layer, 0)
        _calls[layer] = call + 1
        name = (
            f"{_process_side}-rank-{_process_rank}-pp-{_process_pp_rank}-"
            f"cp-{_process_cp_rank}-layer-{layer}-call-{call}.npz"
        )
        path = root / name
        if path.exists():
            raise FileExistsError(f"router trace chunk already exists: {path}")
        temporary = root / f".{name}.{os.getpid()}"
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                ids=ids_array,
                weights=weight_array,
            )
        os.replace(temporary, path)
        _written_files += 1
        _written_bytes += estimate
        if (
            _process_side == "trainer"
            and _process_cp_rank == 0
            and not _provenance_written
            and _owned_layers == _expected_owned_layers
        ):
            _trainer_provenance(root, _owned_layers)
            _provenance_written = True
        _runtime_calls[layer] = _runtime_calls.get(layer, 0) + 1
        source = id_source
        previous_source = _runtime_id_sources.get(layer)
        if previous_source is not None and previous_source != source:
            raise ValueError("router trace ID source changed within one request")
        _runtime_id_sources[layer] = source
        previous_scale = _runtime_scales.get(layer)
        if previous_scale is not None and previous_scale != float(route_output_scale):
            raise ValueError("route output scale changed within one request")
        _runtime_scales[layer] = float(route_output_scale)
        if _process_side == "h200":
            lifecycle_path = _lifecycle_path(root, _process_rank)
            if not lifecycle_path.is_file():
                raise ValueError("H200 router trace request flags were not received")
            lifecycle = json.loads(lifecycle_path.read_text())
            if lifecycle.get("context_entered") is not True:
                raise ValueError("H200 router trace context was not entered")
            lifecycle["router_calls"] = int(lifecycle.get("router_calls", 0)) + 1
            _write_lifecycle(root, lifecycle)
            marker = {
                "session_identity": _active_h200_session,
                "request_identity": _process_request,
                "layer_calls": {
                    str(number): count
                    for number, count in sorted(_runtime_calls.items())
                },
                "id_sources": {
                    str(number): value
                    for number, value in sorted(_runtime_id_sources.items())
                },
                "route_output_scales": {
                    str(number): value
                    for number, value in sorted(_runtime_scales.items())
                },
            }
            marker_path = root / runtime_marker_name
            marker_temporary = root / f".{runtime_marker_name}.{os.getpid()}"
            marker_temporary.write_text(
                json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n"
            )
            os.replace(marker_temporary, marker_path)


def write_router_debug_chunk(
    layer: int,
    router_input: Any,
    router_logits: Any,
    pre_bias_scores: Any,
    selection_scores: Any,
    selection_bias: Any | None = None,
    selection_bias_source_sha256: str | None = None,
    selection_bias_cache_sha256: str | None = None,
    selection_bias_is_static: bool = False,
) -> None:
    """Write one bounded router diagnostic chunk for the first routed layer."""
    global _debug_written, _debug_written_bytes, _debug_written_session
    root = validated_trace_root()
    if root is None or _process_side is None:
        return
    if not router_debug_enabled(layer):
        return
    if diagnostics_enabled() and layer == read_session(root).dense_layers:
        write_hidden_diagnostic(f"layer{layer}_router_input", router_input)
    session = read_session(root)
    with _write_lock:
        session_identity = trace_session_identity(session)
        if _debug_written_session != session_identity:
            _debug_written = False
            _debug_written_bytes = 0
            _debug_written_session = session_identity
        if layer != session.dense_layers or _debug_written:
            return
        limit = int(os.environ.get("PRIME_PARITY_ROUTER_DEBUG_TOKENS", "256"))
        max_bytes = int(os.environ.get("PRIME_PARITY_ROUTER_DEBUG_MAX_BYTES", "50000000"))
        if limit < 1 or max_bytes < 1:
            raise ValueError("router debug limits must be positive")
        torch = __import__("torch")
        input_array = router_input.detach()[:limit].to(dtype=torch.float32, device="cpu").numpy()
        logits_array = router_logits.detach()[:limit].to(dtype=torch.float32, device="cpu").numpy()
        pre_bias_array = pre_bias_scores.detach()[:limit].to(dtype=torch.float32, device="cpu").numpy()
        selection_array = selection_scores.detach()[:limit].to(dtype=torch.float32, device="cpu").numpy()
        bias_array = None
        if selection_bias is not None:
            bias_array = selection_bias.detach().to(dtype=torch.float32, device="cpu").numpy()
        if (
            input_array.ndim != 2
            or logits_array.ndim != 2
            or pre_bias_array.shape != logits_array.shape
            or selection_array.shape != logits_array.shape
        ):
            raise ValueError("router debug arrays have invalid shapes")
        if logits_array.shape[1] != session.experts:
            raise ValueError("router debug logits have the wrong expert count")
        if (
            not np.isfinite(input_array).all()
            or not np.isfinite(logits_array).all()
            or not np.isfinite(pre_bias_array).all()
            or not np.isfinite(selection_array).all()
        ):
            raise ValueError("router debug arrays contain non-finite values")
        estimate = int(
            input_array.nbytes
            + logits_array.nbytes
            + pre_bias_array.nbytes
            + selection_array.nbytes
            + (0 if bias_array is None else bias_array.nbytes)
            + (0 if selection_bias_source_sha256 is None else 64)
            + (0 if selection_bias_cache_sha256 is None else 64)
            + 4096
        )
        if estimate > max_bytes:
            raise ValueError("router debug byte limit exceeded")
        filename = (
            f"{_process_side}-rank-{_process_rank}-pp-{_process_pp_rank}-"
            f"cp-{_process_cp_rank}-layer-{layer}-router-debug.npz"
        )
        path = root / filename
        temporary = root / f".{filename}.{os.getpid()}"
        if path.exists():
            raise FileExistsError(f"router debug chunk already exists: {path}")
        with temporary.open("wb") as stream:
            arrays = dict(
                router_input=input_array,
                router_logits=logits_array,
                pre_bias_scores=pre_bias_array,
                selection_scores=selection_array,
            )
            if bias_array is not None:
                arrays["selection_bias"] = bias_array
                arrays["selection_bias_object_id"] = np.asarray(id(selection_bias), dtype=np.uint64)
                data_ptr = selection_bias.data_ptr() if hasattr(selection_bias, "data_ptr") else 0
                arrays["selection_bias_data_ptr"] = np.asarray(data_ptr, dtype=np.uint64)
                arrays["selection_bias_is_static"] = np.asarray(selection_bias_is_static, dtype=np.bool_)
                if selection_bias_source_sha256 is not None:
                    arrays["selection_bias_source_sha256"] = np.asarray(
                        selection_bias_source_sha256, dtype="<U64"
                    )
                if selection_bias_cache_sha256 is not None:
                    arrays["selection_bias_cache_sha256"] = np.asarray(
                        selection_bias_cache_sha256, dtype="<U64"
                    )
            np.savez(stream, **arrays)
        os.replace(temporary, path)
        _debug_written = True
        _debug_written_bytes += estimate


def router_debug_enabled(layer: int | None) -> bool:
    if layer is None or validated_trace_root() is None or _process_side is None:
        return False
    if _process_side == "h200":
        if _process_rank != 0:
            return False
        return _active_h200_session is not None and _process_request is not None
    return True


def assemble_chunks(root: Path, session: TraceSession) -> tuple[np.ndarray, np.ndarray]:
    if read_session(root) != session:
        raise ValueError("router trace session metadata differs")
    grouped: dict[int, dict[int, list[tuple[int, Path]]]] = {}
    for path in root.glob(f"{session.side}-rank-*-pp-*-cp-*-layer-*-call-*.npz"):
        match = chunk_pattern.match(path.name)
        if match is None:
            raise ValueError(f"invalid router trace chunk name: {path.name}")
        side, _rank, _pp, cp, layer, call = match.groups()
        if side != session.side:
            continue
        layer_number = int(layer)
        grouped.setdefault(layer_number, {}).setdefault(int(cp), []).append(
            (int(call), path)
        )
    ids = np.zeros(
        (session.expected_tokens, session.hidden_layers, session.topk),
        dtype=np.int32,
    )
    weights = np.zeros_like(ids, dtype=np.float32)
    for layer in range(session.dense_layers, session.hidden_layers):
        cp_files = grouped.get(layer)
        if cp_files is None:
            raise ValueError(f"router trace has no data for layer {layer}")
        layer_ids: list[np.ndarray] = []
        layer_weights: list[np.ndarray] = []
        for cp_rank in sorted(cp_files):
            calls = sorted(cp_files[cp_rank])
            if [call for call, _ in calls] != list(range(len(calls))):
                raise ValueError(f"router trace calls are not continuous for layer {layer}")
            cp_ids: list[np.ndarray] = []
            cp_weights: list[np.ndarray] = []
            for _call, path in calls:
                with np.load(path, allow_pickle=False) as values:
                    chunk_ids, chunk_weights = validate_sparse_routes(
                        values["ids"],
                        values["weights"],
                        topk=session.topk,
                        experts=session.experts,
                        sum_tolerance=session.weight_sum_tolerance,
                    )
                cp_ids.append(chunk_ids)
                cp_weights.append(chunk_weights)
            layer_ids.append(np.concatenate(cp_ids, axis=0))
            layer_weights.append(np.concatenate(cp_weights, axis=0))
        combined_ids = np.concatenate(layer_ids, axis=0)
        combined_weights = np.concatenate(layer_weights, axis=0)
        if combined_ids.shape != (session.expected_tokens, session.topk):
            raise ValueError(
                f"router trace layer {layer} has shape {combined_ids.shape}, "
                f"expected {(session.expected_tokens, session.topk)}"
            )
        ids[:, layer, :] = combined_ids
        weights[:, layer, :] = combined_weights
    return ids, weights


def encoded_array(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "data": base64.b64encode(memoryview(array).cast("B")).decode("ascii"),
        "dtype": array.dtype.name,
        "shape": list(array.shape),
    }


def decoded_array(value: dict[str, Any]) -> np.ndarray:
    try:
        raw = base64.b64decode(value["data"], validate=True)
        array = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
        return array.reshape(value["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("encoded router array is invalid") from error


def validate_encoded_diagnostic_input(value: dict[str, Any]) -> tuple[int, int]:
    """Validate shape and base64 size before decoding or device allocation."""
    if not isinstance(value, dict):
        raise ValueError("encoded diagnostic input is not an object")
    data = value.get("data")
    dtype = value.get("dtype")
    shape = value.get("shape")
    if not isinstance(data, str) or len(data) > diagnostic_max_request_bytes:
        raise ValueError("encoded diagnostic input is too large")
    if dtype != "float32" or not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("encoded diagnostic input must be float32 rank two")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in shape):
        raise ValueError("encoded diagnostic input shape is invalid")
    tokens, width = shape
    if tokens < 1 or tokens > 256 or width != diagnostic_hidden_width:
        raise ValueError("encoded diagnostic input shape is outside the diagnostic limit")
    expected_bytes = tokens * width * np.dtype(np.float32).itemsize
    expected_encoded_length = 4 * ((expected_bytes + 2) // 3)
    if len(data) != expected_encoded_length:
        raise ValueError("encoded diagnostic input length is invalid")
    return expected_bytes, expected_encoded_length


def install_vllm_parity_trace() -> bool:
    root = validated_trace_root()
    if root is None:
        return False
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if not getattr(FusedMoERouter, "_prime_parity_trace_installed", False):
        original_select = FusedMoERouter.select_experts

        def traced_select(
            self,
            hidden_states,
            router_logits,
            topk_indices_dtype=None,
            *,
            input_ids=None,
        ):
            route_weights, route_ids = original_select(
                self,
                hidden_states,
                router_logits,
                topk_indices_dtype=topk_indices_dtype,
                input_ids=input_ids,
            )
            layer = getattr(self, "_prime_parity_layer", None)
            if layer is not None and not h200_diagnostic_is_active():
                route_output_scale = getattr(
                    self, "_prime_parity_route_output_scale", None
                )
                write_route_chunk(
                    layer,
                    route_ids,
                    route_weights,
                    route_output_scale=route_output_scale,
                )
                import torch

                pre_bias_scores = (
                    0.5 * torch.tanh(0.5 * router_logits.to(torch.float32)) + 0.5
                )
                selection_scores = pre_bias_scores
                correction_bias = getattr(self, "e_score_correction_bias", None)
                if correction_bias is not None:
                    selection_scores = selection_scores + correction_bias.to(torch.float32)
                write_router_debug_chunk(
                    layer,
                    hidden_states,
                    router_logits,
                    pre_bias_scores,
                    selection_scores,
                )
            return route_weights, route_ids

        FusedMoERouter.select_experts = traced_select
        FusedMoERouter._prime_parity_trace_installed = True

    if getattr(GPUModelRunner, "_prime_parity_trace_installed", False):
        return True
    original_bind = GPUModelRunner._bind_routed_experts_capturer

    def traced_bind(self, capturer):
        result = original_bind(self, capturer)
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter as Router

        for module in self.model.modules():
            if isinstance(module, MoERunner) and isinstance(module.router, Router):
                module.router._prime_parity_layer = module.layer_id
                module.router._prime_parity_route_output_scale = (
                    module.routed_scaling_factor
                )
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        configured = configure_process_trace(
            "h200",
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_world_size(),
        )
        if configured and diagnostics_enabled():
            configure_h200_boundary_diagnostics(self.model)
        return result

    GPUModelRunner._bind_routed_experts_capturer = traced_bind
    GPUModelRunner._prime_parity_trace_installed = True
    return True
