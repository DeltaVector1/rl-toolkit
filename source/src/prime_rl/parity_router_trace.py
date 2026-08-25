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
trainer_provenance_union_name = "trainer-provenance-union.json"
trainer_fixture_source_name = "trainer-fixture-source.npz"
trainer_fixture_manifest_name = "trainer-fixture-source.json"
trainer_bundle_stable_name = "STABLE"
trainer_bundle_slab_bytes = 2_049_028
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
diagnostic_npz_overhead_bytes = 64 * 1024
internal_diagnostic_tokens = 32


def _compact_internal_diagnostic(name: str) -> bool:
    return name.startswith(("layer0_dense_", "layer3_moe_", "layer3_shared_"))


diagnostic_token0_widths = {
    "layer0_normalized_input": 6144,
    "layer0_kv_before_norm": 512,
    "layer0_normalized_kv": 512,
    "layer0_o_proj_input_tp0": 2048,
}
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


@dataclass(frozen=True)
class TrainerBundleCommunication:
    slab: Any | None
    error: str | None


def prepare_trainer_bundle_communication(
    enabled: bool, device: Any
) -> TrainerBundleCommunication | None:
    if not enabled:
        return None
    try:
        import torch

        slab = torch.empty(
            (trainer_bundle_slab_bytes,), dtype=torch.uint8, device=device
        )
        return TrainerBundleCommunication(slab=slab, error=None)
    except BaseException as error:
        return TrainerBundleCommunication(
            slab=None, error=f"trainer bundle slab allocation failed: {error}"[:512]
        )


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


def bootstrap_readiness(
    expected_token: str | None,
    authorization: str | None,
) -> str:
    if expected_token is None:
        return "not_ready"
    if not diagnostic_authorization_valid(authorization, expected_token):
        return "unauthorized"
    return "ready"


def consume_single_use_bootstrap_token(
    expected_token: str | None,
    authorization: str | None,
) -> None:
    if not diagnostic_authorization_valid(authorization, expected_token):
        raise ValueError("parity bootstrap authorization is invalid")


def issue_one_use_bootstrap_token(
    path: Path,
    expected_path: Path,
    expected_uid: int,
) -> str:
    if path != expected_path:
        raise ValueError("parity bootstrap FIFO path is invalid")
    metadata = os.lstat(path)
    if (
        not stat.S_ISFIFO(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != expected_uid
    ):
        raise ValueError("parity bootstrap FIFO type, mode, or owner is invalid")
    token = secrets.token_urlsafe(48)
    payload = (token + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("parity bootstrap FIFO write stopped early")
            written += count
    finally:
        os.close(descriptor)
    return token


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


def diagnostic_capture_preflight(session: TraceSession, side: str) -> dict[str, Any]:
    if side not in {"trainer", "h200"}:
        raise ValueError("diagnostic capture side is invalid")
    limits = diagnostic_limits()
    hidden_names = ["embedding"]
    for layer in range(session.dense_layers):
        hidden_names.extend(
            (
                f"layer{layer}_input",
                f"layer{layer}_attention_raw",
                f"layer{layer}_mlp_raw",
                f"layer{layer}_output",
            )
        )
    hidden_names.append(f"layer{session.dense_layers}_router_input")
    plan = [(name, diagnostic_hidden_width, limits["tokens"]) for name in hidden_names]
    plan.extend((name, width, 1) for name, width in diagnostic_token0_widths.items())
    shared_output_name = (
        "layer3_shared_output_full"
        if side == "trainer"
        else "layer3_shared_output_local_tp0"
    )
    plan.extend(
        (
            ("layer0_dense_input", 6144, internal_diagnostic_tokens),
            ("layer0_dense_gate_up_raw_tp0", 1024, internal_diagnostic_tokens),
            ("layer0_dense_post_silu_tp0", 512, internal_diagnostic_tokens),
            (shared_output_name, 6144, internal_diagnostic_tokens),
            ("layer3_moe_combined", 6144, internal_diagnostic_tokens),
        )
    )
    if side == "h200":
        plan.extend(
            (
                (f"layer{session.dense_layers}_h200_input", diagnostic_hidden_width, limits["tokens"]),
                (f"layer{session.dense_layers}_h200_logits", session.experts, limits["tokens"]),
                (f"layer{session.dense_layers}_h200_scores", session.experts, limits["tokens"]),
                (f"layer{session.dense_layers}_h200_selection_scores", session.experts, limits["tokens"]),
                (f"layer{session.dense_layers}_h200_route_ids", session.topk, limits["tokens"]),
                (f"layer{session.dense_layers}_h200_route_weights", session.topk, limits["tokens"]),
            )
        )
    names = [name for name, _width, _tokens in plan]
    if len(names) != len(set(names)):
        raise ValueError("diagnostic capture plan contains duplicate semantic files")
    estimated_bytes = sum(
        tokens * width * (2 if _compact_internal_diagnostic(_name) else 4)
        + diagnostic_npz_overhead_bytes
        for _name, width, tokens in plan
    )
    if len(plan) > limits["max_files"] or estimated_bytes > limits["max_bytes"]:
        raise ValueError("diagnostic capture plan exceeds the configured file or byte cap")
    return {
        "side": side,
        "files": len(plan),
        "estimated_bytes": estimated_bytes,
        "names": sorted(names),
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


def _trainer_provenance_payload(owned_layers: set[int]) -> dict[str, Any]:
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
    return {
        "schema_version": trace_schema_version,
        "kind": "glm52_policy_parity_trainer_runtime",
        "image_digest": image_digest,
        "host": _trainer_host_name(),
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


def _trainer_provenance(root: Path, owned_layers: set[int]) -> None:
    global _trainer_provenance_record
    payload = _trainer_provenance_payload(owned_layers)
    host = payload["host"]
    filename = f"trainer-runtime-pp-{_process_pp_rank}-host-{host}.json"
    temporary = root / f".{filename}.{os.getpid()}"
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, root / filename)
    _trainer_provenance_record = payload


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
                trainer_provenance_union_name,
                trainer_fixture_source_name,
                trainer_fixture_manifest_name,
                trainer_bundle_stable_name,
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
            diagnostic_capture_preflight(session, "trainer")
            if cp_rank == 0:
                configure_hidden_diagnostics(language_model, layer_start, layer_end)
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
_trainer_provenance_record: dict[str, Any] | None = None
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
    global _owned_layers, _provenance_written, _trainer_provenance_record
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
    _trainer_provenance_record = None
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


def compact_trainer_fixture_source(columns: dict[str, Any]) -> dict[str, np.ndarray] | None:
    root = validated_trace_root()
    if root is None or _process_side != "trainer":
        return None
    session = read_session(root)
    token_ids = np.asarray(columns.get("token_ids"), dtype=np.int64)
    if token_ids.shape != (session.expected_tokens,):
        return None
    if sha256_tokens(token_ids) != session.token_ids_sha256:
        return None
    positions = np.asarray(columns.get("position_ids"), dtype=np.int64)
    selected_mask = np.asarray(columns.get("loss_mask"), dtype=np.bool_)
    full_advantages = np.asarray(columns.get("advantages"), dtype=np.float32)
    if any(
        value.shape != (session.expected_tokens,)
        for value in (positions, selected_mask, full_advantages)
    ):
        raise ValueError("trainer fixture source arrays do not match the trace context")
    if not np.isfinite(full_advantages).all():
        raise ValueError("trainer fixture source advantages are not finite")
    selected_indices = np.flatnonzero(selected_mask).astype(np.int64)
    result = {
        "token_ids": np.ascontiguousarray(token_ids),
        "positions": np.ascontiguousarray(positions),
        "selected_indices": np.ascontiguousarray(selected_indices),
        "advantages": np.ascontiguousarray(full_advantages[selected_indices]),
    }
    if sum(value.nbytes for value in result.values()) > 2 * 1024 * 1024:
        raise ValueError("trainer fixture source exceeds its byte limit")
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B"))


def _content_record(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = sha256_bytes(_canonical_json_bytes(result))
    return result


def _validate_trainer_bundle_inputs(
    gathered: list[dict[str, Any]],
    session: TraceSession,
    trainer_head: str,
    source: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], str]:
    expected_world = _process_pp_size * _process_cp_size
    if len(gathered) != expected_world:
        raise ValueError("trainer provenance gather does not cover the world")
    session_identity = trace_session_identity(session)
    records: list[dict[str, Any]] = []
    compact_source_ranks: list[int] = []
    for rank, item in enumerate(gathered):
        if (
            not isinstance(item, dict)
            or item.get("rank") != rank
            or item.get("session_identity") != session_identity
        ):
            raise ValueError("trainer provenance gather rank or session is invalid")
        record = item.get("provenance")
        expected_cp = rank % _process_cp_size
        if expected_cp == 0:
            if not isinstance(record, dict):
                raise ValueError("trainer CP0 provenance is missing")
            records.append(record)
        elif record is not None:
            raise ValueError("trainer provenance was supplied by a non-CP0 rank")
        if item.get("has_compact_source") is True:
            compact_source_ranks.append(rank)
        elif item.get("has_compact_source") is not False:
            raise ValueError("trainer compact fixture source flag is invalid")
        if item.get("error") is not None:
            raise ValueError(f"trainer bundle rank {rank} failed: {item['error']}")
    records.sort(key=lambda value: int(value.get("pp_rank", -1)))
    if len(records) != _process_pp_size:
        raise ValueError("trainer provenance does not contain every PP stage")
    expected_layers = set(range(session.dense_layers, session.hidden_layers))
    observed_layers: set[int] = set()
    hosts: set[str] = set()
    modules: dict[str, Any] | None = None
    for pp_rank, record in enumerate(records):
        cp_rank = record.get("cp_rank")
        global_rank = record.get("global_rank")
        host = record.get("host")
        if (
            record.get("schema_version") != trace_schema_version
            or
            record.get("kind") != "glm52_policy_parity_trainer_runtime"
            or record.get("pp_rank") != pp_rank
            or cp_rank != 0
            or global_rank != pp_rank * _process_cp_size
            or record.get("pp_size") != _process_pp_size
            or record.get("cp_size") != _process_cp_size
            or record.get("batch_size") != 8
            or not isinstance(host, str)
            or re.fullmatch(r"n[0-9]+", host) is None
            or host in hosts
        ):
            raise ValueError("trainer provenance PP/CP/rank/host mapping is invalid")
        if pp_rank == 0 and host != trainer_head:
            raise ValueError("trainer PP0 host differs from the trainer head")
        hosts.add(host)
        image = record.get("image_digest")
        if not isinstance(image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
            raise ValueError("trainer provenance image digest is invalid")
        current_modules = record.get("modules")
        if not isinstance(current_modules, dict) or not current_modules:
            raise ValueError("trainer provenance module hashes are missing")
        for name, module in current_modules.items():
            if (
                not isinstance(name, str)
                or not isinstance(module, dict)
                or module.get("location") not in {"mounted", "unmounted"}
                or re.fullmatch(r"[0-9a-f]{64}", str(module.get("sha256", ""))) is None
            ):
                raise ValueError("trainer provenance module hash record is invalid")
        if modules is None:
            modules = current_modules
        elif current_modules != modules:
            raise ValueError("trainer provenance module hashes differ")
        owned = record.get("owned_layers")
        if not isinstance(owned, list) or not owned:
            raise ValueError("trainer provenance layer ownership is missing")
        owned_set = set(owned)
        if len(owned_set) != len(owned) or not owned_set <= expected_layers:
            raise ValueError("trainer provenance layer ownership is invalid")
        if observed_layers & owned_set:
            raise ValueError("trainer provenance contains duplicate layer ownership")
        observed_layers.update(owned_set)
    if observed_layers != expected_layers:
        raise ValueError("trainer provenance layer ownership is incomplete")
    if len(compact_source_ranks) != 1:
        raise ValueError("trainer compact fixture source is missing or duplicated")
    source_rank = compact_source_ranks[0]
    expected_source_rank = (_process_pp_size - 1) * _process_cp_size
    if source_rank != expected_source_rank:
        raise ValueError("trainer compact fixture source is not from final PP/CP0")
    expected_arrays = {
        "token_ids": (np.dtype(np.int64), (session.expected_tokens,)),
        "positions": (np.dtype(np.int64), (session.expected_tokens,)),
        "selected_indices": (np.dtype(np.int64), None),
        "advantages": (np.dtype(np.float32), None),
    }
    arrays: dict[str, np.ndarray] = {}
    if set(source) != set(expected_arrays):
        raise ValueError("trainer compact fixture source fields are invalid")
    for name, (dtype, shape) in expected_arrays.items():
        value = np.asarray(source[name])
        if value.dtype != dtype or (shape is not None and value.shape != shape):
            raise ValueError(f"trainer compact fixture source {name} is invalid")
        arrays[name] = np.ascontiguousarray(value)
    selected = arrays["selected_indices"]
    if (
        selected.ndim != 1
        or arrays["advantages"].shape != selected.shape
        or selected.size < 1
        or np.any(selected[1:] <= selected[:-1])
        or selected[0] < 0
        or selected[-1] >= session.expected_tokens
    ):
        raise ValueError("trainer compact fixture selection is invalid")
    if sha256_tokens(arrays["token_ids"]) != session.token_ids_sha256:
        raise ValueError("trainer compact fixture token hash differs from its session")
    if not np.isfinite(arrays["advantages"]).all():
        raise ValueError("trainer compact fixture advantages are not finite")
    if sum(value.nbytes for value in arrays.values()) > 2 * 1024 * 1024:
        raise ValueError("trainer compact fixture source exceeds its byte limit")
    assert modules is not None
    records_hash = sha256_bytes(_canonical_json_bytes(records))
    binding = {
        "session_identity": session_identity,
        "model_revision": session.model_revision,
        "adapter_sha256": session.adapter_sha256,
        "fixture_sha256": session.fixture_sha256,
        "token_ids_sha256": session.token_ids_sha256,
        "records_sha256": records_hash,
        "modules_sha256": sha256_bytes(_canonical_json_bytes(modules)),
    }
    return records, arrays, sha256_bytes(_canonical_json_bytes(binding))


def _write_trainer_bundle(
    root: Path,
    session: TraceSession,
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    binding_sha256: str,
) -> None:
    _validate_root_identity(root)
    finals = {
        "union": root / trainer_provenance_union_name,
        "source": root / trainer_fixture_source_name,
        "manifest": root / trainer_fixture_manifest_name,
        "stable": root / trainer_bundle_stable_name,
    }
    if any(path.exists() or path.is_symlink() for path in finals.values()):
        raise FileExistsError("trainer provenance bundle already exists")
    temporary = {
        name: root / f".{path.name}.{os.getpid()}"
        for name, path in finals.items()
    }
    try:
        with temporary["source"].open("xb") as stream:
            np.savez(stream, **arrays)
        if temporary["source"].stat().st_size > 2 * 1024 * 1024:
            raise ValueError("trainer compact fixture NPZ exceeds its byte limit")
        source_sha256 = _file_sha256(temporary["source"])
        binding = {
            "session_identity": trace_session_identity(session),
            "model_revision": session.model_revision,
            "adapter_sha256": session.adapter_sha256,
            "fixture_sha256": session.fixture_sha256,
            "token_ids_sha256": session.token_ids_sha256,
            "records_sha256": sha256_bytes(_canonical_json_bytes(records)),
            "modules_sha256": sha256_bytes(
                _canonical_json_bytes(records[0]["modules"])
            ),
        }
        if sha256_bytes(_canonical_json_bytes(binding)) != binding_sha256:
            raise ValueError("trainer provenance binding changed before publish")
        array_records = {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        }
        compact_manifest = _content_record(
            {
                "schema_version": trace_schema_version,
                "kind": "glm52_policy_parity_trainer_fixture_source",
                "binding_sha256": binding_sha256,
                "rank": (_process_pp_size - 1) * _process_cp_size,
                "file": trainer_fixture_source_name,
                "file_sha256": source_sha256,
                "arrays": array_records,
            }
        )
        temporary["manifest"].write_bytes(_canonical_json_bytes(compact_manifest))
        manifest_sha256 = _file_sha256(temporary["manifest"])
        union = _content_record(
            {
                "schema_version": trace_schema_version,
                "kind": "glm52_policy_parity_trainer_provenance_union",
                "binding_sha256": binding_sha256,
                "binding": binding,
                "session": session.as_dict(),
                "session_identity": trace_session_identity(session),
                "records": records,
                "fixture_source_manifest_sha256": manifest_sha256,
                "fixture_source_sha256": source_sha256,
            }
        )
        temporary["union"].write_bytes(_canonical_json_bytes(union))
        stable = _content_record(
            {
                "schema_version": trace_schema_version,
                "kind": "glm52_policy_parity_trainer_bundle_stable",
                "binding_sha256": binding_sha256,
                "provenance_union_sha256": _file_sha256(temporary["union"]),
                "fixture_source_manifest_sha256": manifest_sha256,
                "fixture_source_sha256": source_sha256,
            }
        )
        temporary["stable"].write_bytes(_canonical_json_bytes(stable))
        if sum(path.stat().st_size for path in temporary.values()) > 18 * 1024 * 1024:
            raise ValueError("trainer provenance bundle exceeds its byte limit")
        for name in ("source", "manifest", "union", "stable"):
            os.replace(temporary[name], finals[name])
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def _trainer_bundle_slab_views(slab: Any) -> dict[str, Any]:
    import torch

    if (
        slab.dtype != torch.uint8
        or slab.numel() != trainer_bundle_slab_bytes
        or slab.device.type != "cuda"
    ):
        raise ValueError("trainer bundle communication slab is invalid")
    offset = 0

    def take(byte_count: int, dtype: Any) -> Any:
        nonlocal offset
        value = slab[offset : offset + byte_count].view(dtype)
        offset += byte_count
        return value

    views = {
        "local_metadata": take(16 * 1024, torch.uint8),
        "gathered_metadata": take(64 * 16 * 1024, torch.uint8),
        "token_ids": take(32768 * 8, torch.int64),
        "positions": take(32768 * 8, torch.int64),
        "selected_indices": take(32768 * 8, torch.int64),
        "advantages": take(32768 * 4, torch.float32),
        "local_errors": take(1024, torch.uint8),
        "gathered_errors": take(64 * 1024, torch.uint8),
        "status": take(4, torch.int32),
    }
    if offset != trainer_bundle_slab_bytes:
        raise ValueError("trainer bundle communication slab layout differs")
    return views


def publish_trainer_bundle(
    compact_source: dict[str, np.ndarray] | None,
    enabled: bool,
    communication: TrainerBundleCommunication | None = None,
    readiness: Any = None,
) -> None:
    if not enabled:
        return
    import torch.distributed as dist
    import torch

    if not dist.is_available() or not dist.is_initialized():
        raise ValueError("trainer provenance bundle requires distributed training")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if (
        readiness is None
        or readiness.numel() != 2
        or readiness.device.type != "cuda"
    ):
        raise ValueError("trainer bundle readiness storage is invalid")
    communication_error = None
    slab = None
    try:
        if communication is None:
            raise ValueError("trainer bundle communication was not prepared")
        if communication.error is not None:
            raise ValueError(communication.error)
        slab = communication.slab
        if slab is None:
            raise ValueError("trainer bundle communication slab is missing")
        if (
            slab.dtype != torch.uint8
            or slab.numel() != trainer_bundle_slab_bytes
            or slab.device.type != "cuda"
        ):
            raise ValueError("trainer bundle communication slab is invalid")
    except BaseException as error:
        communication_error = str(error)[:512]
    readiness[0] = int(communication_error is not None)
    readiness[1] = 0
    dist.all_reduce(readiness, op=dist.ReduceOp.MAX)
    if int(readiness[0].item()) != 0:
        if communication_error is not None:
            raise RuntimeError("trainer bundle communication is unavailable") from ValueError(
                communication_error
            )
        raise RuntimeError("trainer bundle communication is unavailable on another rank")
    views = None
    view_error = None
    try:
        views = _trainer_bundle_slab_views(slab)
        views["token_ids"].zero_()
        views["positions"].zero_()
        views["selected_indices"].fill_(-1)
        views["advantages"].zero_()
        views["local_errors"].zero_()
        views["status"].zero_()
    except BaseException as error:
        view_error = str(error)[:512]
    readiness[0] = int(view_error is not None)
    readiness[1] = 0
    dist.all_reduce(readiness, op=dist.ReduceOp.MAX)
    if int(readiness[0].item()) != 0:
        if view_error is not None:
            raise RuntimeError("trainer bundle slab setup failed") from ValueError(
                view_error
            )
        raise RuntimeError("trainer bundle slab setup failed on another rank")
    local_metadata = views["local_metadata"]
    gathered_metadata = views["gathered_metadata"]
    token_ids = views["token_ids"]
    positions = views["positions"]
    selected_indices = views["selected_indices"]
    advantages = views["advantages"]
    local_errors = views["local_errors"]
    gathered_errors = views["gathered_errors"]
    status = views["status"]
    errors: list[str] = []
    root: Path | None = None
    session: TraceSession | None = None
    session_identity = ""
    try:
        root = validated_trace_root()
        if root is None:
            raise ValueError("trainer parity trace root is missing")
        if _process_side != "trainer":
            raise ValueError("trainer provenance bundle requires trainer tracing")
        session = read_session(root)
        if session.expected_tokens != 32768:
            raise ValueError("trainer parity session must contain 32768 tokens")
        session_identity = trace_session_identity(session)
    except BaseException as error:
        errors.append(f"root/session preparation failed: {error}"[:512])
    if (
        world_size != 64
        or _process_world_size != 64
        or _process_pp_size != 8
        or _process_cp_size != 8
        or rank != _process_rank
    ):
        errors.append("parity trainer bundle requires exact world64/PP8/CP8 mapping")
    source_rank = 56
    if compact_source is not None and rank != source_rank:
        errors.append("compact fixture source was supplied by a non-final rank")
    local = {
        "rank": rank,
        "session_identity": session_identity,
        "provenance": _trainer_provenance_record,
        "has_compact_source": compact_source is not None,
        "error": "; ".join(errors) or None,
    }
    metadata_limit = 16 * 1024
    try:
        metadata_bytes = _canonical_json_bytes(local)
        if len(metadata_bytes) + 4 > metadata_limit:
            raise ValueError("trainer provenance metadata exceeds 16 KiB")
    except BaseException as error:
        errors.append(f"metadata preparation failed: {error}"[:512])
        safe_error = "; ".join(errors).encode(
            "utf-8", errors="replace"
        )[:2048].decode("utf-8", errors="replace")
        metadata_bytes = json.dumps(
            {
                "rank": rank,
                "session_identity": session_identity,
                "provenance": None,
                "has_compact_source": compact_source is not None,
                "error": safe_error,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    encoded = len(metadata_bytes).to_bytes(4, "little") + metadata_bytes
    encoded += b"\0" * (metadata_limit - len(encoded))
    try:
        local_metadata.copy_(
            torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8))
        )
    except BaseException as error:
        errors.append(f"metadata transfer failed: {error}"[:512])
    dist.all_gather_into_tensor(gathered_metadata, local_metadata)
    if rank == source_rank and not errors:
        try:
            if compact_source is None:
                raise ValueError("final PP/CP0 compact fixture source is missing")
            if set(compact_source) != {
                "token_ids",
                "positions",
                "selected_indices",
                "advantages",
            }:
                raise ValueError("compact fixture source fields differ")
            if not all(
                isinstance(compact_source[name], np.ndarray)
                for name in compact_source
            ):
                raise TypeError("compact fixture source values must be arrays")
            selected_count = int(compact_source["selected_indices"].size)
            if (
                compact_source["token_ids"].shape != (32768,)
                or compact_source["positions"].shape != (32768,)
                or compact_source["selected_indices"].shape != (selected_count,)
                or compact_source["advantages"].shape != (selected_count,)
                or selected_count < 1
                or selected_count > 32768
                or compact_source["token_ids"].dtype != np.int64
                or compact_source["positions"].dtype != np.int64
                or compact_source["selected_indices"].dtype != np.int64
                or compact_source["advantages"].dtype != np.float32
            ):
                raise ValueError("compact fixture source shape or dtype differs")
            source_arrays = {
                name: np.ascontiguousarray(compact_source[name])
                for name in compact_source
            }
            for target, name in (
                (token_ids, "token_ids"),
                (positions, "positions"),
                (selected_indices, "selected_indices"),
                (advantages, "advantages"),
            ):
                source_value = torch.from_numpy(source_arrays[name])
                target[: source_value.numel()].copy_(source_value)
        except BaseException as error:
            errors.append(f"source preparation failed: {error}"[:512])

    error_limit = 1024
    error_bytes = "; ".join(errors).encode("utf-8", errors="replace")[: error_limit - 4]
    error_encoded = len(error_bytes).to_bytes(4, "little") + error_bytes
    error_encoded += b"\0" * (error_limit - len(error_encoded))
    try:
        local_errors.copy_(
            torch.from_numpy(np.frombuffer(error_encoded, dtype=np.uint8))
        )
    except BaseException as error:
        errors.append(f"error transfer failed: {error}"[:512])
    readiness[0] = int(bool(errors))
    readiness[1] = 0
    dist.all_reduce(readiness, op=dist.ReduceOp.MAX)
    any_error = bool(int(readiness[0].item()))
    dist.all_gather_into_tensor(gathered_errors, local_errors)
    if not any_error:
        for value in (token_ids, positions, selected_indices, advantages):
            dist.broadcast(value, src=source_rank)

    failure: BaseException | None = None
    if rank == 0:
        try:
            error_rows = gathered_errors.reshape(
                world_size, error_limit
            ).cpu().numpy()
            rank_errors = []
            for error_rank, row in enumerate(error_rows):
                length = int.from_bytes(row[:4].tobytes(), "little")
                if length > error_limit - 4:
                    raise ValueError("trainer preparation error length is invalid")
                if length:
                    rank_errors.append(
                        f"rank {error_rank}: {row[4 : 4 + length].tobytes().decode('utf-8')}"
                    )
            if any_error or rank_errors:
                raise ValueError("; ".join(rank_errors) or "trainer preparation failed")
            rows = gathered_metadata.reshape(world_size, metadata_limit).cpu().numpy()
            gathered: list[dict[str, Any]] = []
            for row in rows:
                length = int.from_bytes(row[:4].tobytes(), "little")
                if length < 2 or length > metadata_limit - 4:
                    raise ValueError("trainer provenance metadata length is invalid")
                value = json.loads(row[4 : 4 + length].tobytes())
                if not isinstance(value, dict):
                    raise ValueError("trainer provenance metadata is invalid")
                gathered.append(value)
            selected_cpu = selected_indices.cpu().numpy()
            selected_count = int(np.count_nonzero(selected_cpu >= 0))
            if np.any(selected_cpu[:selected_count] < 0) or np.any(
                selected_cpu[selected_count:] != -1
            ):
                raise ValueError("trainer compact selection padding is invalid")
            arrays = {
                "token_ids": token_ids.cpu().numpy(),
                "positions": positions.cpu().numpy(),
                "selected_indices": selected_cpu[:selected_count],
                "advantages": advantages.cpu().numpy()[:selected_count],
            }
            trainer_head = os.environ.get("PRIME_TRAINER_HEAD", "")
            if re.fullmatch(r"n[0-9]+", trainer_head) is None:
                raise ValueError("PRIME_TRAINER_HEAD must identify the trainer head")
            if session is None or root is None:
                raise ValueError("trainer head session or root is unavailable")
            records, checked_arrays, binding = _validate_trainer_bundle_inputs(
                gathered, session, trainer_head, arrays
            )
            _write_trainer_bundle(
                root, session, records, checked_arrays, binding
            )
        except BaseException as error:
            failure = error
    if rank == 0:
        status[0] = int(failure is not None)
    dist.broadcast(status, src=0)
    if int(status.item()) != 0:
        if failure is not None:
            raise RuntimeError("trainer provenance bundle failed") from failure
        raise RuntimeError("trainer provenance bundle failed on trainer head")


def _diagnostic_array(
    value: Any, limit: int, preserve_bfloat16: bool = False
) -> tuple[np.ndarray, str]:
    tensor = value.detach()
    source_dtype = str(getattr(tensor, "dtype", "unknown"))
    if getattr(tensor, "ndim", 0) < 2:
        raise ValueError("diagnostic tensor must have at least two dimensions")
    tensor = tensor.reshape(-1, tensor.shape[-1])[:limit]
    torch = __import__("torch")
    if preserve_bfloat16:
        if tensor.dtype != torch.bfloat16:
            raise TypeError("compact diagnostic input must be BF16")
        tensor = tensor.to(device="cpu").contiguous().view(torch.uint16)
    else:
        tensor = tensor.to(dtype=torch.float32, device="cpu")
    array = np.ascontiguousarray(tensor.numpy())
    if array.ndim != 2 or (
        not preserve_bfloat16 and not np.isfinite(array).all()
    ):
        raise ValueError("diagnostic tensor is invalid")
    return array, source_dtype


def _diagnostic_tensor_metadata(
    value: Any, limit: int, expected_width: int | None, item_size: int = 4
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
    estimated_bytes = token_count * shape[-1] * item_size + diagnostic_npz_overhead_bytes
    return token_count, source_dtype, estimated_bytes


def write_hidden_diagnostic(
    kind: str,
    value: Any,
    *,
    expected_width: int | None = diagnostic_hidden_width,
    token_limit: int | None = None,
    preserve_bfloat16: bool = False,
) -> None:
    """Write one bounded activation capture when explicitly enabled."""
    global _diagnostic_files, _diagnostic_bytes, _diagnostic_session
    root = validated_trace_root()
    if root is None or _process_side is None or not diagnostics_enabled():
        return
    if _process_side == "trainer" and _process_cp_rank != 0:
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
    if token_limit is None:
        token_limit = limits["tokens"]
    if isinstance(token_limit, bool) or not isinstance(token_limit, int):
        raise ValueError("diagnostic token limit must be an integer")
    if token_limit < 1 or token_limit > limits["tokens"]:
        raise ValueError("diagnostic token limit exceeds the configured limit")
    with _write_lock:
        session_identity = trace_session_identity(session)
        if _diagnostic_session != session_identity:
            _diagnostic_files = 0
            _diagnostic_bytes = 0
            _diagnostic_session = session_identity
        token_count, source_dtype, estimated_bytes = _diagnostic_tensor_metadata(
            value, token_limit, expected_width, 2 if preserve_bfloat16 else 4
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
            array, source_dtype = _diagnostic_array(
                value, token_limit, preserve_bfloat16
            )
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


def _h200_router_forward_context(model_runner: Any, num_tokens: int) -> Any:
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
        raise ValueError("H200 diagnostic token count is invalid")
    vllm_config = getattr(model_runner, "vllm_config", None)
    if vllm_config is None:
        raise ValueError("H200 model runner has no vLLM configuration")
    try:
        from vllm.forward_context import set_forward_context

        parameters = inspect.signature(set_forward_context).parameters
    except (ImportError, TypeError, ValueError) as error:
        raise ValueError("vLLM forward context API is unavailable") from error
    required = {"attn_metadata", "vllm_config", "num_tokens", "skip_compiled"}
    if not required.issubset(parameters):
        raise ValueError("vLLM forward context API is incompatible")
    return set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        skip_compiled=True,
    )


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
        with _h200_router_forward_context(model_runner, int(input_array.shape[0])):
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


def _capture_layer0_normalized_input(
    _module: Any, _inputs: Any, output: Any
) -> None:
    if not hasattr(output, "detach"):
        raise ValueError("layer 0 normalized input is not a tensor")
    write_hidden_diagnostic(
        "layer0_normalized_input", output, expected_width=6144, token_limit=1
    )


def _capture_layer0_normalized_kv(
    _module: Any, _inputs: Any, output: Any
) -> None:
    if not hasattr(output, "detach"):
        raise ValueError("layer 0 normalized KV value is not a tensor")
    write_hidden_diagnostic(
        "layer0_normalized_kv", output, expected_width=512, token_limit=1
    )


def _capture_layer0_kv_before_norm(_module: Any, inputs: Any) -> None:
    if not isinstance(inputs, tuple) or not inputs or not hasattr(inputs[0], "detach"):
        raise ValueError("layer 0 KV input is not a tensor")
    write_hidden_diagnostic(
        "layer0_kv_before_norm", inputs[0], expected_width=512, token_limit=1
    )


def _capture_layer0_o_proj_input(_module: Any, inputs: Any) -> None:
    if not isinstance(inputs, tuple) or not inputs or not hasattr(inputs[0], "detach"):
        raise ValueError("layer 0 output projection input is not a tensor")
    value = inputs[0]
    width = int(value.shape[-1])
    if width == 16384:
        value = value[..., :2048]
    elif width != 2048:
        raise ValueError("layer 0 output projection input has an invalid width")
    write_hidden_diagnostic(
        "layer0_o_proj_input_tp0", value, expected_width=2048, token_limit=1
    )


def _install_layer0_token_diagnostics(layer: Any) -> None:
    input_norm = getattr(layer, "input_layernorm", None)
    attention = getattr(layer, "self_attn", None)
    attention_modules = getattr(attention, "mla_attn", attention)
    kv_norm = getattr(attention_modules, "kv_a_layernorm", None)
    output_projection = getattr(attention_modules, "o_proj", None)
    if any(
        module is None or not hasattr(module, "register_forward_hook")
        for module in (input_norm, kv_norm, output_projection)
    ) or any(
        not hasattr(module, "register_forward_pre_hook")
        for module in (kv_norm, output_projection)
    ):
        raise ValueError("layer 0 token diagnostics cannot find attention boundaries")
    _diagnostic_hooks.append(
        input_norm.register_forward_hook(_capture_layer0_normalized_input)
    )
    _diagnostic_hooks.append(
        kv_norm.register_forward_pre_hook(_capture_layer0_kv_before_norm)
    )
    _diagnostic_hooks.append(
        kv_norm.register_forward_hook(_capture_layer0_normalized_kv)
    )
    _diagnostic_hooks.append(
        output_projection.register_forward_pre_hook(_capture_layer0_o_proj_input)
    )


def _diagnostic_output_tensor(output: Any, label: str) -> Any:
    if hasattr(output, "detach"):
        return output
    if isinstance(output, tuple) and output and hasattr(output[0], "detach"):
        return output[0]
    raise ValueError(f"{label} output is not a tensor")


def _capture_named_output(kind: str, width: int, output: Any) -> None:
    write_hidden_diagnostic(
        kind,
        _diagnostic_output_tensor(output, kind),
        expected_width=width,
        token_limit=internal_diagnostic_tokens,
        preserve_bfloat16=True,
    )


def _capture_named_input(kind: str, width: int, inputs: Any) -> None:
    if not isinstance(inputs, tuple) or not inputs or not hasattr(inputs[0], "detach"):
        raise ValueError(f"{kind} input is not a tensor")
    write_hidden_diagnostic(
        kind,
        inputs[0],
        expected_width=width,
        token_limit=internal_diagnostic_tokens,
        preserve_bfloat16=True,
    )


def _install_trainer_dense_diagnostics(module: Any) -> None:
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    if any(item is None for item in (gate, up, down)):
        raise ValueError("layer 0 cannot find trainer dense projections")
    projection_state: dict[str, Any] = {}
    _diagnostic_hooks.append(
        module.register_forward_pre_hook(
            lambda _module, inputs: _capture_named_input(
                "layer0_dense_input", 6144, inputs
            )
        )
    )
    _diagnostic_hooks.append(
        gate.register_forward_hook(
            lambda _module, _inputs, output: projection_state.__setitem__(
                "gate", _diagnostic_output_tensor(output, "layer 0 dense gate")
            )
        )
    )

    def capture_up(_module: Any, _inputs: Any, output: Any) -> None:
        gate_output = projection_state.pop("gate", None)
        if gate_output is None:
            raise ValueError("layer 0 dense gate output was not captured")
        up_output = _diagnostic_output_tensor(output, "layer 0 dense up")
        if gate_output.shape[-1] != 4096 or up_output.shape[-1] != 4096:
            raise ValueError("layer 0 trainer dense projection width is not 4096")
        packed = __import__("torch").cat(
            (gate_output[..., :512], up_output[..., :512]), dim=-1
        )
        _capture_named_output("layer0_dense_gate_up_raw_tp0", 1024, packed)

    _diagnostic_hooks.append(up.register_forward_hook(capture_up))

    def capture_post_silu(_module: Any, inputs: Any) -> None:
        value = _diagnostic_output_tensor(inputs, "layer 0 dense post-SiLU")
        if value.shape[-1] != 4096:
            raise ValueError("layer 0 trainer post-SiLU width is not 4096")
        _capture_named_output("layer0_dense_post_silu_tp0", 512, value[..., :512])

    _diagnostic_hooks.append(down.register_forward_pre_hook(capture_post_silu))


def _install_h200_dense_diagnostics(module: Any) -> None:
    gate_up = getattr(module, "gate_up_proj", None)
    activation = getattr(module, "act_fn", None)
    down = getattr(module, "down_proj", None)
    if any(item is None for item in (gate_up, activation, down)):
        raise ValueError("layer 0 cannot find H200 dense projections")
    _diagnostic_hooks.append(
        module.register_forward_pre_hook(
            lambda _module, inputs: _capture_named_input(
                "layer0_dense_input", 6144, inputs
            )
        )
    )

    def capture_gate_up(_module: Any, _inputs: Any, output: Any) -> None:
        packed = _diagnostic_output_tensor(output, "layer 0 packed gate/up")
        if packed.shape[-1] != 1024:
            raise ValueError("layer 0 packed gate/up width is not 1024")
        _capture_named_output("layer0_dense_gate_up_raw_tp0", 1024, packed)

    _diagnostic_hooks.append(gate_up.register_forward_hook(capture_gate_up))
    _diagnostic_hooks.append(
        activation.register_forward_hook(
            lambda _module, _inputs, output: _capture_named_output(
                "layer0_dense_post_silu_tp0", 512, output
            )
        )
    )


def _install_h200_moe_diagnostics(module: Any) -> None:
    shared = getattr(module, "shared_experts", None)
    if shared is None:
        raise ValueError("layer 3 H200 diagnostics cannot find the shared expert")
    _diagnostic_hooks.append(
        shared.register_forward_hook(
            lambda _module, _inputs, output: _capture_named_output(
                "layer3_shared_output_local_tp0", 6144, output
            )
        )
    )
    _diagnostic_hooks.append(
        module.register_forward_hook(
            lambda _module, _inputs, output: _capture_named_output(
                "layer3_moe_combined", 6144, output
            )
        )
    )


def configure_hidden_diagnostics(
    language_model: Any, layer_start: int, layer_end: int
) -> None:
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
    if embed is not None and layer_start <= 0 < layer_end:
        _diagnostic_hooks.append(
            embed.register_forward_hook(_capture_embedding)
        )
    items = layers.items() if hasattr(layers, "items") else enumerate(layers)
    for raw_layer, layer in items:
        layer_number = int(getattr(layer, "layer_idx", raw_layer))
        if layer_number < layer_start or layer_number >= layer_end or layer_number > 3:
            continue
        if layer_number == 3:
            mlp = getattr(layer, "mlp", None)
            shared = getattr(mlp, "shared_expert", None)
            if mlp is None or shared is None:
                raise ValueError("trainer layer 3 MoE diagnostic modules are missing")
            _diagnostic_hooks.append(
                shared.register_forward_hook(
                    lambda _module, _inputs, output: _capture_named_output(
                        "layer3_shared_output_full", 6144, output
                    )
                )
            )
            _diagnostic_hooks.append(
                mlp.register_forward_hook(
                    lambda _module, _inputs, output: _capture_named_output(
                        "layer3_moe_combined", 6144, output
                    )
                )
            )
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
        if layer_number == 0:
            _install_layer0_token_diagnostics(layer)
            if mlp is None:
                raise ValueError("trainer layer 0 dense MLP is missing")
            _install_trainer_dense_diagnostics(mlp)


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
        if layer_number in {0, 1, 2, 3}:
            selected[layer_number] = layer
    if set(selected) != {0, 1, 2, 3}:
        raise ValueError("H200 diagnostics require local logical layers 0 through 3")
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
        if layer_number == 0:
            _install_layer0_token_diagnostics(layer)
            _install_h200_dense_diagnostics(mlp)
    _install_h200_moe_diagnostics(getattr(selected[3], "mlp", None))


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
    if _process_side == "trainer" and _process_cp_rank != 0:
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
