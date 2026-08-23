#!/usr/bin/env python3
"""Build and verify a direct GLM trainer-to-inference parity record."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import os
import secrets
import shutil
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


schema_version = 1
required_targets = (
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
)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Profile:
    path: Path
    values: dict[str, Any]

    def get(self, name: str) -> Any:
        return self.values[name]


def load_profile(path: Path) -> Profile:
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    exact = {
        "schema_version": schema_version,
        "quantization": "native_fp8",
        "model_architecture": "GlmMoeDsaForCausalLM",
        "context_length": 32768,
        "h200_scoring_context_length": 32769,
        "prompt_length": 16384,
        "max_output_length": 16384,
        "vocab_size": 154880,
        "hidden_layers": 78,
        "dense_layers": 3,
        "experts": 256,
        "experts_per_token": 8,
        "router_scale": 2.5,
        "lora_rank": 32,
        "lora_alpha": 64,
        "dppo_mask_low": 0.2,
        "dppo_mask_high": 0.2,
        "max_unmasked_mismatch_kl": 0.1,
        "max_masked_mismatch_kl": 0.5,
        "max_masked_fraction": 0.06,
        "max_router_weight_sum_error": 0.005,
    }
    for name, expected in exact.items():
        if values.get(name) != expected:
            raise ValueError(f"profile {name} must be {expected!r}")
    if tuple(values.get("lora_targets", ())) != required_targets:
        raise ValueError("profile does not contain the exact full LoRA target list")
    if (
        values.get("router_weight_delta_status")
        != "provisional_until_first_base_and_adapter_capture"
    ):
        raise ValueError("router weight delta status must remain provisional")
    router_weight_limit = values.get("max_router_weight_abs_delta")
    if (
        isinstance(router_weight_limit, bool)
        or not isinstance(router_weight_limit, (int, float))
        or router_weight_limit <= 0
    ):
        raise ValueError("provisional router weight delta limit must be positive")
    if values["prompt_length"] + values["max_output_length"] != values["context_length"]:
        raise ValueError("prompt and output lengths do not fill the context")
    if not isinstance(values.get("prime_rl_revision"), str) or len(values["prime_rl_revision"]) != 40:
        raise ValueError("profile needs one full Prime-RL revision")
    if values.get("max_fixture_age_hours", 0) <= 0:
        raise ValueError("max_fixture_age_hours must be positive")
    return Profile(path=path, values=values)


def parse_time(value: str | None) -> dt.datetime:
    if value is None:
        return dt.datetime.now(dt.timezone.utc)
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("created time has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def make_fixture(
    profile: Profile,
    output: Path,
    seed: int | None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    seed = secrets.randbits(64) if seed is None else seed
    length = profile.get("context_length")
    vocab_size = profile.get("vocab_size")
    generator = np.random.Generator(np.random.PCG64(seed))
    token_ids = generator.integers(256, vocab_size, size=length, dtype=np.int64)
    positions = np.arange(length, dtype=np.int64)
    selected = np.arange(profile.get("prompt_length"), length, dtype=np.int64)
    advantages = np.where((token_ids[selected] ^ selected) & 1, 1.0, -1.0).astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "token_ids": token_ids,
        "positions": positions,
        "selected_indices": selected,
        "advantages": advantages,
    }
    entries: dict[str, Any] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(output / filename, array, allow_pickle=False)
        entries[name] = {
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": file_sha256(output / filename),
        }
    manifest = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_fixture",
        "created_utc": parse_time(created_utc).isoformat(),
        "seed": seed,
        "profile_sha256": file_sha256(profile.path),
        "context_length": length,
        "prompt_length": profile.get("prompt_length"),
        "max_output_length": profile.get("max_output_length"),
        "arrays": entries,
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def load_manifest(path: Path, kind: str) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != schema_version or manifest.get("kind") != kind:
        raise ValueError(f"invalid {kind} manifest in {path}")
    for entry in manifest.get("arrays", {}).values():
        array_path = path / entry["file"]
        if file_sha256(array_path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {array_path}")
    return manifest


def fixture_digest(path: Path) -> str:
    manifest = load_manifest(path, "glm52_policy_parity_fixture")
    return hashlib.sha256(json_bytes(manifest)).hexdigest()


def load_array(root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    try:
        entry = manifest["arrays"][name]
    except KeyError as error:
        raise ValueError(f"missing array {name} in {root}") from error
    return np.load(root / entry["file"], allow_pickle=False)


def check_fresh(profile: Profile, fixture: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("profile_sha256") != file_sha256(profile.path):
        raise ValueError("fixture profile checksum does not match")
    created = dt.datetime.fromisoformat(manifest["created_utc"])
    if created.tzinfo is None:
        raise ValueError("fixture time has no timezone")
    age = dt.datetime.now(dt.timezone.utc) - created.astimezone(dt.timezone.utc)
    if age.total_seconds() < -300:
        raise ValueError("fixture creation time is in the future")
    if age > dt.timedelta(hours=profile.get("max_fixture_age_hours")):
        raise ValueError(f"fixture is stale: {fixture}")


def expected_router_shape(profile: Profile) -> tuple[int, int, int]:
    return (
        profile.get("context_length"),
        profile.get("hidden_layers"),
        profile.get("experts_per_token"),
    )


def validate_logprobs(profile: Profile, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    expected = profile.get("max_output_length")
    if values.shape != (expected,):
        raise ValueError(f"selected logprobs shape must be {(expected,)}, got {values.shape}")
    if not np.isfinite(values).all() or np.any(values > 0):
        raise ValueError("selected logprobs must be finite and no greater than zero")
    return values


def validate_router_ids(profile: Profile, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != expected_router_shape(profile):
        raise ValueError(f"router IDs shape must be {expected_router_shape(profile)}, got {values.shape}")
    if values.dtype.kind not in "iu":
        raise ValueError("router IDs must use an integer dtype")
    sparse = values[:, profile.get("dense_layers") :, :]
    if sparse.size and (sparse.min() < 0 or sparse.max() >= profile.get("experts")):
        raise ValueError("router ID is outside the model expert range")
    if sparse.size and not np.any(sparse):
        raise ValueError("sparse router IDs are zero-filled")
    if sparse.size and np.any(np.diff(np.sort(sparse, axis=-1), axis=-1) == 0):
        raise ValueError("a sparse token routes to the same expert more than once")
    return np.ascontiguousarray(values.astype(np.uint8, copy=False))


def validate_router_weights(profile: Profile, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != expected_router_shape(profile):
        raise ValueError(
            f"router weights shape must be {expected_router_shape(profile)}, got {values.shape}"
        )
    sparse = values[:, profile.get("dense_layers") :, :]
    if not np.isfinite(sparse).all() or np.any(sparse <= 0):
        raise ValueError("sparse router weights must be finite and positive")
    max_sum_error = float(np.max(np.abs(sparse.sum(axis=-1) - 1.0)))
    limit = profile.get("max_router_weight_sum_error")
    if max_sum_error > limit:
        raise ValueError(f"router weight sum error is {max_sum_error:.6f}, limit is {limit:.6f}")
    return np.ascontiguousarray(values)

def validate_route_output_scales(
    profile: Profile,
    trainer_manifest: dict[str, Any],
    h200_manifest: dict[str, Any],
) -> float:
    scales = []
    for name, manifest in (("trainer", trainer_manifest), ("H200", h200_manifest)):
        if (
            manifest.get("router_weight_provenance")
            != "selected_weights_before_route_output_scale"
        ):
            raise ValueError(f"{name} router weight provenance is invalid")
        scale = manifest.get("route_output_scale")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{name} route output scale is missing or invalid")
        scales.append(float(scale))
    if scales[0] != scales[1]:
        raise ValueError("trainer and H200 route output scales differ")
    expected = profile.get("router_scale")
    if (
        not isinstance(expected, (int, float))
        or not math.isfinite(expected)
        or expected <= 0
    ):
        raise ValueError("profile route output scale is missing or invalid")
    if scales[0] != expected:
        raise ValueError("trace route output scale differs from profile")
    return scales[0]


def record_trace(
    profile: Profile,
    fixture: Path,
    output: Path,
    side: str,
    selected_logprobs_path: Path,
    router_ids_path: Path,
    router_weights_path: Path | None,
    model_revision: str,
    adapter_sha256: str,
    image: str,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    if side not in {"trainer", "h200"}:
        raise ValueError("side must be trainer or h200")
    if len(model_revision) != 40:
        raise ValueError("model revision must be a full 40-character revision")
    if len(adapter_sha256) != 64 or any(c not in "0123456789abcdef" for c in adapter_sha256):
        raise ValueError("adapter checksum must be 64 lowercase hex characters")
    logprobs = validate_logprobs(profile, np.load(selected_logprobs_path, allow_pickle=False))
    router_ids = validate_router_ids(profile, np.load(router_ids_path, allow_pickle=False))
    router_weights = None
    if router_weights_path is not None:
        router_weights = validate_router_weights(
            profile, np.load(router_weights_path, allow_pickle=False)
        )
    if router_weights is None:
        raise ValueError(f"{side} trace must include router weights")
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "selected_logprobs": logprobs,
        "router_ids": router_ids,
    }
    if router_weights is not None:
        arrays["router_weights"] = router_weights
    entries: dict[str, Any] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(output / filename, array, allow_pickle=False)
        entries[name] = {
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": file_sha256(output / filename),
        }
    manifest = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_trace",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "side": side,
        "fixture_sha256": fixture_digest(fixture),
        "profile_sha256": file_sha256(profile.path),
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "image": image,
        "lora_targets": list(required_targets),
        "quantization": profile.get("quantization"),
        "lora_rank": profile.get("lora_rank"),
        "lora_alpha": profile.get("lora_alpha"),
        "context_length": profile.get("context_length"),
        "route_output_scale": profile.get("router_scale"),
        "router_weight_provenance": "selected_weights_before_route_output_scale",
        "max_output_length": profile.get("max_output_length"),
        "arrays": entries,
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def safe_mean(values: np.ndarray) -> float:
    return float(values.mean(dtype=np.float64)) if values.size else 0.0


def response_logprob(entry: Any, token_id: int, index: int) -> float:
    if index == 0 and entry is None:
        return 0.0
    if not isinstance(entry, dict) or not entry:
        raise ValueError(f"prompt logprob entry {index} is missing")
    value = None
    for key in (str(token_id), token_id, f"token_id:{token_id}"):
        if key in entry:
            value = entry[key]
            break
    if value is None and len(entry) == 1:
        value = next(iter(entry.values()))
    if isinstance(value, dict):
        value = value.get("logprob")
    elif hasattr(value, "logprob"):
        value = value.logprob
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prompt logprob entry {index} has no numeric value for token {token_id}")
    result = float(value)
    if not math.isfinite(result) or result > 0:
        raise ValueError(f"prompt logprob entry {index} is invalid")
    return result


def graph_sampling_params(seed: int) -> dict[str, Any]:
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "prompt_logprobs": 0,
        "logprobs": 1,
        "extra_args": {"prime_exact_target_prompt_logprobs": True},
        "routed_experts_prompt_start": 0,
        "seed": seed,
        "skip_special_tokens": False,
    }


def trace_sampling_params(seed: int, trace_session: dict[str, Any]) -> dict[str, Any]:
    trace_session_id = hashlib.sha256(
        json.dumps(
            trace_session, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    result = graph_sampling_params(seed)
    result["extra_args"].update(
        {
            "prime_router_trace": True,
            "prime_router_trace_session": trace_session_id,
            "prime_router_trace_request": secrets.token_hex(16),
        }
    )
    return result


def prompt_logprobs(response: dict[str, Any], token_ids: np.ndarray) -> np.ndarray:
    entries = response.get("prompt_logprobs")
    if not isinstance(entries, list) or len(entries) != token_ids.size:
        count = len(entries) if isinstance(entries, list) else 0
        raise ValueError(
            f"H200 returned {count} prompt logprobs, expected {token_ids.size}"
        )
    return np.asarray(
        [
            response_logprob(entry, int(token_ids[index]), index)
            for index, entry in enumerate(entries)
        ],
        dtype=np.float32,
    )


def inference_path_metrics(
    profile: Profile,
    advantages: np.ndarray,
    graph_logprobs: np.ndarray,
    eager_logprobs: np.ndarray,
) -> dict[str, Any]:
    graph = np.asarray(graph_logprobs, dtype=np.float64)
    eager = np.asarray(eager_logprobs, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    if graph.shape != eager.shape or graph.shape != advantages.shape:
        raise ValueError("graph and eager scoring inputs have different shapes")
    if not np.isfinite(graph).all() or not np.isfinite(eager).all():
        raise ValueError("graph or eager scoring has a nonfinite logprob")
    log_ratio = eager - graph
    with np.errstate(over="ignore", invalid="ignore"):
        mismatch_kl = np.exp(log_ratio) - log_ratio - 1.0
    if not np.isfinite(mismatch_kl).all():
        raise ValueError("graph and eager mismatch KL is not finite")
    probability_delta = np.exp(eager) - np.exp(graph)
    masked = np.where(
        advantages > 0,
        probability_delta > profile.get("dppo_mask_high"),
        probability_delta < -profile.get("dppo_mask_low"),
    )
    masked_fraction = float(masked.mean())
    masked_kl = safe_mean(mismatch_kl[masked])
    unmasked_kl = safe_mean(mismatch_kl[~masked])
    absolute_delta = np.abs(log_ratio)
    checks = {
        "unmasked_mismatch_kl": unmasked_kl
        <= profile.get("max_unmasked_mismatch_kl"),
        "masked_mismatch_kl": masked_kl
        <= profile.get("max_masked_mismatch_kl"),
        "masked_fraction": masked_fraction <= profile.get("max_masked_fraction"),
    }
    return {
        "passed": all(checks.values()),
        "selected_tokens": int(graph.size),
        "mean_abs_logprob_delta": float(absolute_delta.mean()),
        "p95_abs_logprob_delta": float(np.quantile(absolute_delta, 0.95)),
        "max_abs_logprob_delta": float(absolute_delta.max()),
        "unmasked_mismatch_kl": unmasked_kl,
        "masked_mismatch_kl": masked_kl,
        "masked_fraction": masked_fraction,
        "checks": checks,
    }


def request_json(
    url: str,
    *,
    body: dict[str, Any] | None,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Accept": "application/json", "Connection": "close"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, separators=(",", ":")).encode()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode(errors="replace")
        raise ValueError(f"H200 request failed with HTTP {error.code}: {detail}") from error
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("H200 response is not JSON") from error
    if not isinstance(result, dict):
        raise ValueError("H200 response is not an object")
    return result


def capture_h200(
    profile: Profile,
    fixture: Path,
    output: Path,
    base_url: str,
    model: str,
    model_revision: str,
    adapter_sha256: str,
    image: str,
    api_key_env: str | None,
    timeout: float,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    base = base_url.rstrip("/").removesuffix("/v1")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and api_key is None:
        raise ValueError(f"environment variable {api_key_env} is not set")
    cards = request_json(
        f"{base}/v1/models", body=None, api_key=api_key, timeout=min(timeout, 120)
    ).get("data", [])
    card = next((item for item in cards if isinstance(item, dict) and item.get("id") == model), None)
    if card is None:
        raise ValueError(f"H200 does not serve model {model!r}")
    if card.get("max_model_len") != profile.get("h200_scoring_context_length"):
        raise ValueError(
            "H200 direct scoring needs max_model_len 32769 so it can prefill all "
            "32768 fixture tokens and return one output token"
        )
    token_ids = load_array(fixture, fixture_manifest, "token_ids").astype(np.int64)
    token_ids_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(token_ids)).cast("B")).hexdigest()
    trace_session = {
        "schema_version": schema_version,
        "side": "h200",
        "fixture_sha256": fixture_digest(fixture),
        "token_ids_sha256": token_ids_sha256,
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "image": image,
        "expected_tokens": profile.get("context_length"),
        "hidden_layers": profile.get("hidden_layers"),
        "dense_layers": profile.get("dense_layers"),
        "topk": profile.get("experts_per_token"),
        "experts": profile.get("experts"),
        "route_output_scale": profile.get("router_scale"),
        "weight_sum_tolerance": profile.get("max_router_weight_sum_error"),
    }
    seed = int(fixture_manifest["seed"] & 0x7FFFFFFF)
    common_body = {
        "model": model,
        "token_ids": token_ids.tolist(),
    }
    graph_body = {
        **common_body,
        "cache_salt": f"{fixture_digest(fixture)}:graph",
        "sampling_params": graph_sampling_params(seed),
    }
    graph_response = request_json(
        f"{base}/inference/v1/generate",
        body=graph_body,
        api_key=api_key,
        timeout=timeout,
    )
    graph_full_logprobs = prompt_logprobs(graph_response, token_ids)
    reset = request_json(
        f"{base}/parity/router-trace/reset",
        body=trace_session,
        api_key=api_key,
        timeout=min(timeout, 120),
    )
    if reset != trace_session:
        raise ValueError("H200 router trace reset returned different session metadata")
    eager_sampling_params = trace_sampling_params(seed, trace_session)
    eager_body = {
        **common_body,
        "cache_salt": f"{fixture_digest(fixture)}:eager",
        "sampling_params": eager_sampling_params,
    }
    eager_response = request_json(
        f"{base}/inference/v1/generate",
        body=eager_body,
        api_key=api_key,
        timeout=timeout,
    )
    eager_full_logprobs = prompt_logprobs(eager_response, token_ids)
    choices = eager_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("H200 response needs exactly one choice")
    encoded = choices[0].get("routed_experts")
    if not isinstance(encoded, dict):
        raise ValueError("H200 did not return routed expert IDs")
    try:
        router_ids = np.frombuffer(
            base64.b64decode(encoded["data"], validate=True),
            dtype=np.dtype(encoded["dtype"]),
        ).reshape(encoded["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("H200 routed expert payload is invalid") from error
    if int(encoded.get("start", 0)) != 0:
        raise ValueError("H200 routed expert payload does not start at token zero")
    context = profile.get("context_length")
    if router_ids.ndim != 3 or router_ids.shape[0] < context:
        raise ValueError(f"H200 routed expert payload is too short: {router_ids.shape}")
    router_ids = validate_router_ids(profile, router_ids[:context])
    route_trace = request_json(
        f"{base}/parity/router-trace",
        body=None,
        api_key=api_key,
        timeout=timeout,
    )
    if route_trace.get("session") != trace_session:
        raise ValueError("H200 router trace used different session metadata")
    runtime_marker = route_trace.get("runtime_marker")
    expected_trace_request = eager_sampling_params["extra_args"][
        "prime_router_trace_request"
    ]
    if not isinstance(runtime_marker, dict) or runtime_marker.get(
        "request_identity"
    ) != expected_trace_request:
        raise ValueError("H200 router trace has no matching runtime marker")
    try:
        traced_ids_entry = route_trace["router_ids"]
        traced_weights_entry = route_trace["router_weights"]
        traced_ids = np.frombuffer(
            base64.b64decode(traced_ids_entry["data"], validate=True),
            dtype=np.dtype(traced_ids_entry["dtype"]),
        ).reshape(traced_ids_entry["shape"])
        observed_weights = np.frombuffer(
            base64.b64decode(traced_weights_entry["data"], validate=True),
            dtype=np.dtype(traced_weights_entry["dtype"]),
        ).reshape(traced_weights_entry["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("H200 router weight trace payload is invalid") from error
    traced_ids = validate_router_ids(profile, traced_ids)
    require_exact_router_ids(router_ids, traced_ids)
    router_weights = validate_router_weights(profile, observed_weights)
    selected_indices = load_array(fixture, fixture_manifest, "selected_indices")
    graph_selected_logprobs = validate_logprobs(
        profile, graph_full_logprobs[selected_indices]
    )
    eager_selected_logprobs = validate_logprobs(
        profile, eager_full_logprobs[selected_indices]
    )
    advantages = load_array(fixture, fixture_manifest, "advantages")
    graph_eager_guard = inference_path_metrics(
        profile,
        advantages,
        graph_selected_logprobs,
        eager_selected_logprobs,
    )
    if not graph_eager_guard["passed"]:
        raise ValueError(
            "graph and eager scoring failed the existing parity limits: "
            + json.dumps(graph_eager_guard, sort_keys=True)
        )
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "full_logprobs": graph_full_logprobs,
        "selected_logprobs": graph_selected_logprobs,
        "eager_full_logprobs": eager_full_logprobs,
        "eager_selected_logprobs": eager_selected_logprobs,
        "router_ids": router_ids,
        "router_weights": router_weights,
    }
    entries: dict[str, Any] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(output / filename, array, allow_pickle=False)
        entries[name] = {
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": file_sha256(output / filename),
        }
    manifest = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_h200_capture",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fixture_sha256": fixture_digest(fixture),
        "profile_sha256": file_sha256(profile.path),
        "base_url": base_url,
        "model": model,
        "graph_request_id": graph_response.get("request_id"),
        "eager_request_id": eager_response.get("request_id"),
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "image": image,
        "lora_targets": list(required_targets),
        "route_output_scale": profile.get("router_scale"),
        "router_weight_provenance": "selected_weights_before_route_output_scale",
        "graph_eager_guard": graph_eager_guard,
        "router_runtime_marker": runtime_marker,
        "phases": {
            "graph": {
                "fixture_sha256": fixture_digest(fixture),
                "model": model,
                "model_revision": model_revision,
                "adapter_sha256": adapter_sha256,
                "mode": "normal_cuda_graph",
            },
            "eager_trace": {
                "fixture_sha256": fixture_digest(fixture),
                "model": model,
                "model_revision": model_revision,
                "adapter_sha256": adapter_sha256,
                "mode": "force_eager_skip_compiled",
            },
        },
        "sources": {
            "trainer_reference_logprobs": "normal_cuda_graph_request",
            "h200_trace_logprobs": "normal_cuda_graph_request",
            "router_ids": "isolated_eager_trace_request",
            "router_weights": "isolated_eager_trace_request",
        },
        "arrays": entries,
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def validate_h200_capture_sources(manifest: dict[str, Any]) -> None:
    expected_sources = {
        "trainer_reference_logprobs": "normal_cuda_graph_request",
        "h200_trace_logprobs": "normal_cuda_graph_request",
        "router_ids": "isolated_eager_trace_request",
        "router_weights": "isolated_eager_trace_request",
    }
    if manifest.get("sources") != expected_sources:
        raise ValueError("H200 capture output sources are missing or mixed")
    guard = manifest.get("graph_eager_guard")
    if not isinstance(guard, dict) or guard.get("passed") is not True:
        raise ValueError("H200 graph and eager scoring did not pass parity limits")
    runtime_marker = manifest.get("router_runtime_marker")
    if not isinstance(runtime_marker, dict) or not runtime_marker.get("layer_calls"):
        raise ValueError("H200 router runtime marker is missing")
    id_sources = runtime_marker.get("id_sources")
    if not isinstance(id_sources, dict) or not id_sources:
        raise ValueError("H200 router runtime marker ID sources are missing")
    if any(source != "modular_actual" for source in id_sources.values()):
        raise ValueError("H200 router IDs are not from the executed MoE path")
    scale = manifest.get("route_output_scale")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("H200 route output scale is missing or invalid")
    if (
        manifest.get("router_weight_provenance")
        != "selected_weights_before_route_output_scale"
    ):
        raise ValueError("H200 router weight provenance is invalid")
    scales = runtime_marker.get("route_output_scales")
    if not isinstance(scales, dict) or not scales:
        raise ValueError("H200 route output scales are missing")
    if any(value != scale for value in scales.values()):
        raise ValueError("H200 route output scale differs from capture")
    phases = manifest.get("phases")
    if not isinstance(phases, dict):
        raise ValueError("H200 capture phases are missing")
    graph = phases.get("graph")
    eager = phases.get("eager_trace")
    if not isinstance(graph, dict) or not isinstance(eager, dict):
        raise ValueError("H200 capture phase identity is missing")
    for field in ("fixture_sha256", "model", "model_revision", "adapter_sha256"):
        if graph.get(field) != eager.get(field):
            raise ValueError(f"H200 graph and eager phase {field} values differ")
    if graph.get("mode") != "normal_cuda_graph":
        raise ValueError("H200 graph phase mode is invalid")
    if eager.get("mode") != "force_eager_skip_compiled":
        raise ValueError("H200 eager trace phase mode is invalid")


def require_exact_router_ids(
    executed_ids: np.ndarray, observed_ids: np.ndarray
) -> None:
    if executed_ids.shape != observed_ids.shape:
        raise ValueError("executed and observed router ID shapes differ")
    if not np.array_equal(executed_ids, observed_ids):
        raise ValueError("executed and observed router IDs differ")


def select_first_trainer_router_call(router_trace: Path, output: Path) -> None:
    """Select one identical fixture pass from a multi-microbatch trainer trace."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(router_trace / "session.json", output / "session.json")
    chunks = sorted(router_trace.glob("trainer-rank-*-pp-*-cp-*-layer-*-call-*.npz"))
    first_calls = [path for path in chunks if path.name.endswith("-call-0.npz")]
    if not first_calls:
        raise ValueError("trainer router trace has no first microbatch")
    for path in chunks:
        first_path = output / (path.name.rsplit("-call-", 1)[0] + "-call-0.npz")
        if path.name.endswith("-call-0.npz"):
            shutil.copy2(path, first_path)
            continue
        reference_path = router_trace / first_path.name
        if not reference_path.is_file():
            raise ValueError(f"trainer router trace is missing {reference_path.name}")
        with np.load(reference_path, allow_pickle=False) as reference, np.load(
            path, allow_pickle=False
        ) as candidate:
            for name in ("ids", "weights"):
                if not np.array_equal(reference[name], candidate[name]):
                    raise ValueError(
                        f"trainer router trace microbatch differs for {path.name}"
                    )


def capture_trainer(
    profile: Profile,
    fixture: Path,
    token_exports: Path,
    router_trace: Path,
    output: Path,
    model_revision: str,
    adapter_sha256: str,
    image: str,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    token_ids = load_array(fixture, fixture_manifest, "token_ids").astype(np.int64)
    expected_token_hash = hashlib.sha256(
        memoryview(np.ascontiguousarray(token_ids)).cast("B")
    ).hexdigest()
    source_root = Path(__file__).resolve().parents[1] / "source" / "src"
    sys.path.insert(0, str(source_root))
    from prime_rl.parity_router_trace import assemble_chunks, read_session

    session = read_session(router_trace)
    expected = {
        "side": "trainer",
        "fixture_sha256": fixture_digest(fixture),
        "token_ids_sha256": expected_token_hash,
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "image": image,
        "expected_tokens": profile.get("context_length"),
        "hidden_layers": profile.get("hidden_layers"),
        "dense_layers": profile.get("dense_layers"),
        "topk": profile.get("experts_per_token"),
        "experts": profile.get("experts"),
        "route_output_scale": profile.get("router_scale"),
        "weight_sum_tolerance": profile.get("max_router_weight_sum_error"),
    }
    for name, value in expected.items():
        if getattr(session, name) != value:
            raise ValueError(f"trainer router trace {name} differs from the parity request")
    with tempfile.TemporaryDirectory(prefix="glm52-policy-parity-router-") as temporary:
        first_router_trace = Path(temporary) / "first"
        select_first_trainer_router_call(router_trace, first_router_trace)
        router_ids, router_weights = assemble_chunks(first_router_trace, session)
    matching_records: list[dict[str, Any]] = []
    for path in sorted(token_exports.rglob("rank_*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("schema_version") != schema_version:
                raise ValueError(f"token export schema differs in {path}")
            if record.get("token_ids") == token_ids.tolist():
                matching_records.append(record)
    if not matching_records:
        raise ValueError(
            "expected at least one exact fixture token export, found none"
        )
    first_record = matching_records[0]
    for duplicate in matching_records[1:]:
        for name in (
            "token_ids",
            "position_ids",
            "loss_mask",
            "advantages",
            "inference_logprobs",
            "trainer_logprobs",
            "sequence_lengths",
        ):
            if duplicate.get(name) != first_record.get(name):
                raise ValueError(
                    "duplicate fixture token exports differ in "
                    f"{name}; parity cannot select one microbatch"
                )
    full_logprobs = np.asarray(matching_records[0].get("trainer_logprobs"), dtype=np.float32)
    if full_logprobs.shape != token_ids.shape:
        raise ValueError(
            f"trainer token logprobs shape is {full_logprobs.shape}, expected {token_ids.shape}"
        )
    selected_indices = load_array(fixture, fixture_manifest, "selected_indices")
    selected_logprobs = validate_logprobs(profile, full_logprobs[selected_indices])
    with tempfile.TemporaryDirectory(prefix="glm52-policy-parity-trainer-") as temporary:
        temp = Path(temporary)
        np.save(temp / "selected_logprobs.npy", selected_logprobs, allow_pickle=False)
        np.save(temp / "router_ids.npy", router_ids, allow_pickle=False)
        np.save(temp / "router_weights.npy", router_weights, allow_pickle=False)
        return record_trace(
            profile,
            fixture,
            output,
            "trainer",
            temp / "selected_logprobs.npy",
            temp / "router_ids.npy",
            temp / "router_weights.npy",
            model_revision,
            adapter_sha256,
            image,
        )


def make_trainer_reference(
    profile: Profile,
    fixture: Path,
    h200_capture: Path,
    output: Path,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    capture_manifest = load_manifest(
        h200_capture, "glm52_policy_parity_h200_capture"
    )
    validate_h200_capture_sources(capture_manifest)
    if capture_manifest.get("fixture_sha256") != fixture_digest(fixture):
        raise ValueError("H200 capture used another fixture")
    if capture_manifest.get("profile_sha256") != file_sha256(profile.path):
        raise ValueError("H200 capture used another profile")
    input_ids = load_array(fixture, fixture_manifest, "token_ids").astype(np.int64)
    inference_logprobs = load_array(
        h200_capture, capture_manifest, "full_logprobs"
    ).astype(np.float32)
    if inference_logprobs.shape != input_ids.shape:
        raise ValueError("H200 full logprobs do not align with fixture tokens")
    selected = load_array(fixture, fixture_manifest, "selected_indices").astype(np.int64)
    selected_advantages = load_array(fixture, fixture_manifest, "advantages").astype(
        np.float32
    )
    loss_mask = np.zeros(input_ids.shape, dtype=np.bool_)
    loss_mask[selected] = True
    loss_mask[0] = False
    advantages = np.zeros(input_ids.shape, dtype=np.float32)
    advantages[selected] = selected_advantages
    payload = {
        "schema_version": schema_version,
        "fixture_sha256": fixture_digest(fixture),
        "profile_sha256": file_sha256(profile.path),
        "model_revision": capture_manifest.get("model_revision"),
        "adapter_sha256": capture_manifest.get("adapter_sha256"),
        "image": capture_manifest.get("image"),
        "input_ids": input_ids.tolist(),
        "inference_logprobs": inference_logprobs.tolist(),
        "sequence_lengths": [profile.get("context_length")],
        "loss_mask": loss_mask.tolist(),
        "advantages": advantages.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(json_bytes(payload))
    return {
        "output": str(output),
        "fixture_sha256": payload["fixture_sha256"],
        "tokens": int(input_ids.size),
        "selected_tokens": int(loss_mask.sum()),
        "model_revision": payload["model_revision"],
        "adapter_sha256": payload["adapter_sha256"],
    }


def prepare_trainer_router_trace(
    profile: Profile,
    fixture: Path,
    h200_capture: Path,
    output: Path,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    capture_manifest = load_manifest(
        h200_capture, "glm52_policy_parity_h200_capture"
    )
    validate_h200_capture_sources(capture_manifest)
    if capture_manifest.get("fixture_sha256") != fixture_digest(fixture):
        raise ValueError("H200 capture used another fixture")
    if capture_manifest.get("profile_sha256") != file_sha256(profile.path):
        raise ValueError("H200 capture used another profile")
    token_ids = load_array(fixture, fixture_manifest, "token_ids").astype(np.int64)
    token_ids_sha256 = hashlib.sha256(
        memoryview(np.ascontiguousarray(token_ids)).cast("B")
    ).hexdigest()
    source_root = Path(__file__).resolve().parents[1] / "source" / "src"
    sys.path.insert(0, str(source_root))
    from prime_rl.parity_router_trace import TraceSession, write_session

    trace_session = TraceSession(
        side="trainer",
        fixture_sha256=fixture_digest(fixture),
        token_ids_sha256=token_ids_sha256,
        model_revision=capture_manifest["model_revision"],
        adapter_sha256=capture_manifest["adapter_sha256"],
        image=profile.get("trainer_image"),
        expected_tokens=profile.get("context_length"),
        hidden_layers=profile.get("hidden_layers"),
        dense_layers=profile.get("dense_layers"),
        topk=profile.get("experts_per_token"),
        experts=profile.get("experts"),
        route_output_scale=profile.get("router_scale"),
        weight_sum_tolerance=profile.get("max_router_weight_sum_error"),
    )
    output.mkdir(parents=True, exist_ok=False)
    write_session(output, trace_session)
    return trace_session.as_dict()


def stage_trainer_artifacts(
    profile: Profile,
    fixture: Path,
    h200_capture: Path,
    output: Path,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    capture_manifest = load_manifest(
        h200_capture, "glm52_policy_parity_h200_capture"
    )
    validate_h200_capture_sources(capture_manifest)
    if capture_manifest.get("fixture_sha256") != fixture_digest(fixture):
        raise ValueError("H200 capture used another fixture")
    if capture_manifest.get("profile_sha256") != file_sha256(profile.path):
        raise ValueError("H200 capture used another profile")
    if output.is_symlink():
        raise ValueError("trainer stage output must not be a symlink")
    output.mkdir(parents=False, exist_ok=True)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("trainer stage output must be an empty directory")
    reference = output / "trainer-reference.json"
    router_trace = output / "trainer-router-trace"
    make_trainer_reference(profile, fixture, h200_capture, reference)
    prepare_trainer_router_trace(profile, fixture, h200_capture, router_trace)
    manifest = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_trainer_stage",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile_sha256": file_sha256(profile.path),
        "fixture_sha256": fixture_digest(fixture),
        "h200_capture_manifest_sha256": file_sha256(
            h200_capture / "manifest.json"
        ),
        "route_output_scale": capture_manifest["route_output_scale"],
        "router_weight_provenance": capture_manifest[
            "router_weight_provenance"
        ],
        "files": {
            "trainer-reference.json": file_sha256(reference),
            "trainer-router-trace/session.json": file_sha256(
                router_trace / "session.json"
            ),
        },
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def validate_stage_paths(
    operations_path: Path,
    fixture: Path,
    h200_capture: Path,
    output: Path,
) -> None:
    with operations_path.open("rb") as stream:
        operations = tomllib.load(stream)
    expected = operations.get("parity", {}).get("glm52")
    if not isinstance(expected, dict):
        raise ValueError("operations config has no GLM parity paths")
    actual = {
        "fixture": str(fixture),
        "h200_capture": str(h200_capture),
        "trainer_stage": str(output),
    }
    if actual != expected:
        raise ValueError(
            f"GLM parity staging paths differ from operations config: {actual}"
        )


def compare(profile: Profile, fixture: Path, trainer: Path, h200: Path, output: Path | None) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    trainer_manifest = load_manifest(trainer, "glm52_policy_parity_trace")
    h200_manifest = load_manifest(h200, "glm52_policy_parity_h200_capture")
    validate_h200_capture_sources(h200_manifest)
    if trainer_manifest.get("side") != "trainer":
        raise ValueError("trainer trace side is not trainer")
    expected_fixture = fixture_digest(fixture)
    for name, manifest in (("trainer", trainer_manifest), ("h200", h200_manifest)):
        if manifest.get("fixture_sha256") != expected_fixture:
            raise ValueError(f"{name} trace used another fixture")
        if manifest.get("profile_sha256") != file_sha256(profile.path):
            raise ValueError(f"{name} trace used another profile")
        if tuple(manifest.get("lora_targets", ())) != required_targets:
            raise ValueError(f"{name} trace has the wrong LoRA targets")
    for field in ("model_revision", "adapter_sha256"):
        if trainer_manifest.get(field) != h200_manifest.get(field):
            raise ValueError(f"trace {field} values differ")
    validate_route_output_scales(profile, trainer_manifest, h200_manifest)

    trainer_logprobs = validate_logprobs(
        profile, load_array(trainer, trainer_manifest, "selected_logprobs")
    ).astype(np.float64)
    inference_logprobs = validate_logprobs(
        profile, load_array(h200, h200_manifest, "selected_logprobs")
    ).astype(np.float64)
    trainer_ids = validate_router_ids(profile, load_array(trainer, trainer_manifest, "router_ids"))
    h200_ids = validate_router_ids(profile, load_array(h200, h200_manifest, "router_ids"))
    trainer_weights = validate_router_weights(
        profile, load_array(trainer, trainer_manifest, "router_weights")
    )
    h200_weights = validate_router_weights(
        profile, load_array(h200, h200_manifest, "router_weights")
    )
    dense_layers = profile.get("dense_layers")
    trainer_sparse_ids = trainer_ids[:, dense_layers:, :]
    h200_sparse_ids = h200_ids[:, dense_layers:, :]
    route_exact_fraction = float(np.mean(trainer_sparse_ids == h200_sparse_ids))
    route_exact = bool(np.array_equal(trainer_sparse_ids, h200_sparse_ids))

    advantages = load_array(fixture, fixture_manifest, "advantages").astype(np.float64)
    log_ratio = trainer_logprobs - inference_logprobs
    with np.errstate(over="ignore", invalid="ignore"):
        importance_ratio = np.exp(log_ratio)
        mismatch_kl = importance_ratio - log_ratio - 1.0
    if not np.isfinite(mismatch_kl).all():
        raise ValueError("mismatch KL is not finite")
    probability_delta = np.exp(trainer_logprobs) - np.exp(inference_logprobs)
    masked = np.where(
        advantages > 0,
        probability_delta > profile.get("dppo_mask_high"),
        probability_delta < -profile.get("dppo_mask_low"),
    )
    masked_fraction = float(masked.mean())
    masked_kl = safe_mean(mismatch_kl[masked])
    unmasked_kl = safe_mean(mismatch_kl[~masked])
    delta = trainer_logprobs - inference_logprobs
    trainer_sparse_weights = trainer_weights[:, dense_layers:, :]
    h200_sparse_weights = h200_weights[:, dense_layers:, :]
    trainer_router_weight_sum_max_error = float(
        np.max(np.abs(trainer_sparse_weights.sum(axis=-1) - 1.0))
    )
    h200_router_weight_sum_max_error = float(
        np.max(np.abs(h200_sparse_weights.sum(axis=-1) - 1.0))
    )
    router_weight_abs_delta = np.abs(
        trainer_sparse_weights.astype(np.float64) - h200_sparse_weights.astype(np.float64)
    )
    router_weight_max_abs_delta = float(np.max(router_weight_abs_delta))
    checks = {
        "unmasked_mismatch_kl": unmasked_kl <= profile.get("max_unmasked_mismatch_kl"),
        "masked_mismatch_kl": masked_kl <= profile.get("max_masked_mismatch_kl"),
        "masked_fraction": masked_fraction <= profile.get("max_masked_fraction"),
        "router_ids_exact": route_exact,
        "trainer_router_weight_sums": trainer_router_weight_sum_max_error
        <= profile.get("max_router_weight_sum_error"),
        "h200_router_weight_sums": h200_router_weight_sum_max_error
        <= profile.get("max_router_weight_sum_error"),
    }
    provisional_checks = {
        "router_weight_delta": router_weight_max_abs_delta
        <= profile.get("max_router_weight_abs_delta")
    }
    report = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_report",
        "passed": all(checks.values()),
        "fixture_sha256": expected_fixture,
        "profile_sha256": file_sha256(profile.path),
        "model_revision": trainer_manifest["model_revision"],
        "adapter_sha256": trainer_manifest["adapter_sha256"],
        "inference_logprob_source": "normal_cuda_graph_request",
        "inference_router_source": "isolated_eager_trace_request",
        "graph_eager_guard": h200_manifest["graph_eager_guard"],
        "selected_tokens": int(trainer_logprobs.size),
        "unmasked_mismatch_kl": unmasked_kl,
        "masked_mismatch_kl": masked_kl,
        "masked_fraction": masked_fraction,
        "mean_abs_logprob_delta": float(np.mean(np.abs(delta))),
        "p95_abs_logprob_delta": float(np.quantile(np.abs(delta), 0.95)),
        "max_abs_logprob_delta": float(np.max(np.abs(delta))),
        "router_id_exact_fraction": route_exact_fraction,
        "trainer_router_weight_sum_max_error": trainer_router_weight_sum_max_error,
        "h200_router_weight_sum_max_error": h200_router_weight_sum_max_error,
        "router_weight_mean_abs_delta": float(np.mean(router_weight_abs_delta)),
        "router_weight_p99_abs_delta": float(np.quantile(router_weight_abs_delta, 0.99)),
        "router_weight_max_abs_delta": router_weight_max_abs_delta,
        "thresholds": {
            "max_unmasked_mismatch_kl": profile.get("max_unmasked_mismatch_kl"),
            "max_masked_mismatch_kl": profile.get("max_masked_mismatch_kl"),
            "max_masked_fraction": profile.get("max_masked_fraction"),
            "max_router_weight_abs_delta": profile.get("max_router_weight_abs_delta"),
            "max_router_weight_sum_error": profile.get("max_router_weight_sum_error"),
        },
        "checks": checks,
        "provisional_checks": provisional_checks,
        "router_weight_delta_status": profile.get("router_weight_delta_status"),
    }
    if output is not None:
        output.write_bytes(json_bytes(report))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/qualification/glm52-policy-parity.toml"),
    )
    commands = result.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("output", type=Path)
    fixture.add_argument("--seed", type=int)
    fixture.add_argument("--created-utc")
    h200 = commands.add_parser("capture-h200")
    h200.add_argument("--fixture", type=Path, required=True)
    h200.add_argument("--base-url", required=True)
    h200.add_argument("--model", required=True)
    h200.add_argument("--model-revision", required=True)
    h200.add_argument("--adapter-sha256", required=True)
    h200.add_argument("--image", required=True)
    h200.add_argument("--api-key-env")
    h200.add_argument("--timeout", type=float, default=1800.0)
    h200.add_argument("output", type=Path)
    trainer = commands.add_parser("capture-trainer")
    trainer.add_argument("--fixture", type=Path, required=True)
    trainer.add_argument("--token-exports", type=Path, required=True)
    trainer.add_argument("--router-trace", type=Path, required=True)
    trainer.add_argument("--model-revision", required=True)
    trainer.add_argument("--adapter-sha256", required=True)
    trainer.add_argument("--image", required=True)
    trainer.add_argument("output", type=Path)
    reference = commands.add_parser("make-trainer-reference")
    reference.add_argument("--fixture", type=Path, required=True)
    reference.add_argument("--h200-capture", type=Path, required=True)
    reference.add_argument("output", type=Path)
    trace = commands.add_parser("prepare-trainer-trace")
    trace.add_argument("--fixture", type=Path, required=True)
    trace.add_argument("--h200-capture", type=Path, required=True)
    trace.add_argument("output", type=Path)
    stage = commands.add_parser("stage-trainer")
    stage.add_argument(
        "--operations",
        type=Path,
        default=Path("configs/operations.toml"),
    )
    stage.add_argument("--fixture", type=Path, required=True)
    stage.add_argument("--h200-capture", type=Path, required=True)
    stage.add_argument("output", type=Path)
    record = commands.add_parser("record")
    record.add_argument("--fixture", type=Path, required=True)
    record.add_argument("--side", choices=("trainer", "h200"), required=True)
    record.add_argument("--selected-logprobs", type=Path, required=True)
    record.add_argument("--router-ids", type=Path, required=True)
    record.add_argument("--router-weights", type=Path)
    record.add_argument("--model-revision", required=True)
    record.add_argument("--adapter-sha256", required=True)
    record.add_argument("--image", required=True)
    record.add_argument("output", type=Path)
    check = commands.add_parser("compare")
    check.add_argument("--fixture", type=Path, required=True)
    check.add_argument("--trainer", type=Path, required=True)
    check.add_argument("--h200", type=Path, required=True)
    check.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        profile = load_profile(args.profile)
        if args.command == "fixture":
            result = make_fixture(profile, args.output, args.seed, args.created_utc)
        elif args.command == "capture-h200":
            result = capture_h200(
                profile,
                args.fixture,
                args.output,
                args.base_url,
                args.model,
                args.model_revision,
                args.adapter_sha256,
                args.image,
                args.api_key_env,
                args.timeout,
            )
        elif args.command == "capture-trainer":
            result = capture_trainer(
                profile,
                args.fixture,
                args.token_exports,
                args.router_trace,
                args.output,
                args.model_revision,
                args.adapter_sha256,
                args.image,
            )
        elif args.command == "make-trainer-reference":
            result = make_trainer_reference(
                profile,
                args.fixture,
                args.h200_capture,
                args.output,
            )
        elif args.command == "prepare-trainer-trace":
            result = prepare_trainer_router_trace(
                profile,
                args.fixture,
                args.h200_capture,
                args.output,
            )
        elif args.command == "stage-trainer":
            validate_stage_paths(
                args.operations,
                args.fixture,
                args.h200_capture,
                args.output,
            )
            result = stage_trainer_artifacts(
                profile,
                args.fixture,
                args.h200_capture,
                args.output,
            )
        elif args.command == "record":
            result = record_trace(
                profile,
                args.fixture,
                args.output,
                args.side,
                args.selected_logprobs,
                args.router_ids,
                args.router_weights,
                args.model_revision,
                args.adapter_sha256,
                args.image,
            )
        else:
            result = compare(profile, args.fixture, args.trainer, args.h200, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(args.command == "compare" and not result["passed"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
