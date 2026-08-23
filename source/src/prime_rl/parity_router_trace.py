from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import json
import os
import re
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
    scores = np.reciprocal(
        np.float32(1.0) + np.exp(-logits, dtype=np.float32),
        dtype=np.float32,
    )
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
    path = root / session_name
    if path.exists():
        current = TraceSession.from_dict(json.loads(path.read_text()))
        if current != session:
            raise ValueError("router trace session does not match the existing session")
        return
    temporary = root / f".{session_name}.{os.getpid()}"
    temporary.write_text(json.dumps(session.as_dict(), sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_session(root: Path) -> TraceSession:
    return TraceSession.from_dict(json.loads((root / session_name).read_text()))


def clear_trace(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for path in root.iterdir():
        if (
            path.name in {session_name, runtime_marker_name, trainer_provenance_name}
            or path.name.startswith("trainer-runtime-pp-")
            or chunk_pattern.match(path.name)
            or lifecycle_pattern.match(path.name)
        ):
            if not path.is_file():
                raise ValueError(f"router trace path is not a file: {path}")
            path.unlink()


def configure_trainer_trace(
    config: Any, parallel_dims: Any, world: Any, model: Any = None
) -> bool:
    root = trace_root()
    if root is None:
        return False
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
_active_h200_session: str | None = None
_runtime_calls: dict[int, int] = {}
_runtime_id_sources: dict[int, str] = {}
_runtime_scales: dict[int, float] = {}
_calls: dict[int, int] = {}
_written_files = 0
_written_bytes = 0
_owned_layers: set[int] = set()
_provenance_written = False
_expected_owned_layers: set[int] = set()
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
    root = trace_root()
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
    if side == "h200" and rank != 0:
        _process_side = None
        return False
    _process_side = side
    _process_session = None
    _process_request = None
    _active_h200_session = None
    _calls.clear()
    global _written_files, _written_bytes, _owned_layers, _provenance_written
    _written_files = 0
    _written_bytes = 0
    _owned_layers = set()
    _provenance_written = False
    _runtime_calls.clear()
    _runtime_id_sources.clear()
    _runtime_scales.clear()
    return True


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
    root = trace_root()
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
    root = trace_root()
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
    root = trace_root()
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
    root = trace_root()
    if root is None or _process_side is None:
        return
    if _process_side == "h200" and _active_h200_session is None:
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


def install_vllm_parity_trace() -> bool:
    root = trace_root()
    if root is None:
        return False
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if not getattr(FusedMoERouter, "_prime_parity_trace_installed", False):
        original_select = FusedMoERouter.select_experts

        def traced_select(self, *args, **kwargs):
            route_weights, route_ids = original_select(self, *args, **kwargs)
            layer = getattr(self, "_prime_parity_layer", None)
            if layer is not None:
                route_output_scale = getattr(
                    self, "_prime_parity_route_output_scale", None
                )
                write_route_chunk(
                    layer,
                    route_ids,
                    route_weights,
                    route_output_scale=route_output_scale,
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

        configure_process_trace(
            "h200",
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_world_size(),
        )
        return result

    GPUModelRunner._bind_routed_experts_capturer = traced_bind
    GPUModelRunner._prime_parity_trace_installed = True
    return True
