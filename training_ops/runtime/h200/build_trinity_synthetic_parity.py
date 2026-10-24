"""Build a fresh, deterministic Trinity packed-versus-individual reference.

The input to this builder is generated in this file.  It does not read a
rollout record or a rollout cache.  Five base-model prompt-logprob requests
are sent to the H200 endpoint, then the exact token, logprob, and routed
expert rows are written in both packed and padded individual forms.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SEQUENCE_LENGTHS = (4096, 5632, 6656, 7680, 8704)
TOTAL_TOKENS = sum(SEQUENCE_LENGTHS)
SEQUENCE_COUNT = len(SEQUENCE_LENGTHS)
DEFAULT_BASE_MODEL = "/data/trinity-thinking-fp8"
DEFAULT_SEED = "trinity-synthetic-packed-parity-v1"
SOURCE_KIND = "synthetic"
TOPOLOGY = {"cp": 8, "pp": 8, "ep": 8}
SAMPLING_PARAMS = {
    "max_tokens": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "prompt_logprobs": 1,
}
PARITY_GATES = {
    "max_unmasked_mismatch_kl": 0.1,
    "max_masked_mismatch_kl": 0.5,
    "max_masked_fraction": 0.06,
}
DIAGNOSTIC_NAME = "trinity_h200_synthetic_packed_individual_parity"
ROUTE_CONSTRUCTION = "byte_concat(individual_routes)"
PP_TRANSFER_SPARSE_INDICES = False
VOCAB_SIZE = 151936
PADDING_TOKEN_ID = 0
MAX_ROUTE_BYTES = 512 * 1024 * 1024
DENSE_LAYERS = 6
EXPERTS = 256
_ROUTE_FIELDS = {"data", "dtype", "shape", "start"}
_REFERENCE_ROUTE_FIELDS = {
    "schema_version",
    "encoding",
    "dtype",
    "shape",
    "dense_layers",
    "experts",
    "sha256",
    "data",
}


class SyntheticParityError(ValueError):
    """The synthetic reference or H200 evidence is unsafe to use."""


def json_sha256(value: Any) -> str:
    """Return a stable digest for one JSON value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _token(seed: str, sequence_index: int, position: int) -> int:
    # Hash each position independently.  This keeps the stream deterministic
    # without a global PRNG state and avoids short repeated token windows.
    payload = f"trinity-synthetic-token-v1\0{seed}\0{sequence_index}\0{position}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % VOCAB_SIZE


def _validate_diversity(tokens: Sequence[int], sequence_index: int) -> None:
    if len(set(tokens)) < 64:
        raise SyntheticParityError(
            f"synthetic sequence {sequence_index} does not contain enough token diversity"
        )
    if len(tokens) >= 16:
        windows = [tuple(tokens[index : index + 16]) for index in range(len(tokens) - 15)]
        if len(set(windows)) < len(windows) - max(1, len(windows) // 50):
            raise SyntheticParityError(
                f"synthetic sequence {sequence_index} contains repeated token windows"
            )


def make_synthetic_sequences(
    seed: str = DEFAULT_SEED,
    lengths: Sequence[int] = SEQUENCE_LENGTHS,
) -> list[dict[str, Any]]:
    """Create fresh deterministic token sequences with diversity checks."""
    if not isinstance(seed, str) or not seed:
        raise SyntheticParityError("synthetic seed must be a non-empty string")
    if tuple(lengths) != SEQUENCE_LENGTHS:
        raise SyntheticParityError(
            f"synthetic sequence lengths must be exactly {list(SEQUENCE_LENGTHS)}"
        )
    result = []
    for sequence_index, length in enumerate(SEQUENCE_LENGTHS):
        tokens = [_token(seed, sequence_index, position) for position in range(length)]
        _validate_diversity(tokens, sequence_index)
        result.append(
            {
                "index": sequence_index,
                "input_ids": tokens,
                "length": length,
                "sha256": json_sha256(tokens),
            }
        )
    return result


def build_token_sequences(seed: str = DEFAULT_SEED) -> list[list[int]]:
    """Return only token IDs for callers that do not need sequence metadata."""
    return [list(item["input_ids"]) for item in make_synthetic_sequences(seed)]


def validate_source_kind(value: Mapping[str, Any]) -> None:
    """Reject records that claim an external or cached source."""
    if not isinstance(value, Mapping) or value.get("source_kind") != SOURCE_KIND:
        raise SyntheticParityError("artifact source_kind must be synthetic")
    if any("rollout" in str(key).lower() for key in value):
        raise SyntheticParityError("synthetic artifact cannot contain rollout fields")


def _normalise_sequences(value: Sequence[Any]) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SyntheticParityError("sequences must be a list")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            tokens = item.get("input_ids")
        else:
            tokens = item
        if (
            not isinstance(tokens, list)
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise SyntheticParityError(f"sequence {index} has invalid input_ids")
        result.append({"index": index, "input_ids": list(tokens), "length": len(tokens)})
    if [item["length"] for item in result] != list(SEQUENCE_LENGTHS):
        raise SyntheticParityError("sequences do not have the required lengths")
    for index, item in enumerate(result):
        _validate_diversity(item["input_ids"], index)
        item["sha256"] = json_sha256(item["input_ids"])
    return result


def make_request_descriptions(
    sequences: Sequence[Any],
    base_model: str = DEFAULT_BASE_MODEL,
) -> list[dict[str, Any]]:
    """Build the five exact base-model serialized generate requests."""
    if not isinstance(base_model, str) or not base_model.strip():
        raise SyntheticParityError("base model must be a non-empty string")
    records = _normalise_sequences(sequences)
    return [
        {
            "model": base_model,
            "token_ids": list(record["input_ids"]),
            "sampling_params": dict(SAMPLING_PARAMS),
        }
        for record in records
    ]


def _flatten_prompt_logprobs(raw: Any, input_ids: Sequence[int]) -> list[float]:
    if not isinstance(raw, list) or len(raw) != len(input_ids):
        raise SyntheticParityError("prompt logprob count does not match token count")
    result: list[float] = []
    for index, token_id in enumerate(input_ids):
        entry = raw[index]
        if index == 0 and entry is None:
            result.append(0.0)
            continue
        if not isinstance(entry, Mapping):
            raise SyntheticParityError(f"prompt logprob entry {index} is missing")
        candidates = [
            candidate
            for key, candidate in entry.items()
            if key == token_id or str(key) == str(token_id)
        ]
        if len(candidates) != 1 or not isinstance(candidates[0], Mapping):
            raise SyntheticParityError(
                f"prompt logprob entry {index} is not aligned with token {token_id}"
            )
        value = candidates[0].get("logprob")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SyntheticParityError(f"prompt logprob entry {index} is not numeric")
        value = float(value)
        if not math.isfinite(value):
            raise SyntheticParityError(f"prompt logprob entry {index} is not finite")
        result.append(value)
    return result


def _route_raw(route: Mapping[str, Any], token_count: int | None = None) -> tuple[dict[str, Any], bytes]:
    is_reference_route = isinstance(route, Mapping) and set(route) == _REFERENCE_ROUTE_FIELDS
    if not isinstance(route, Mapping) or (
        (is_reference_route and set(route) != _REFERENCE_ROUTE_FIELDS)
        or (not is_reference_route and set(route) != _ROUTE_FIELDS)
    ):
        raise SyntheticParityError("routed experts have an invalid schema")
    shape = route.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(item) is not int or item <= 0 for item in shape)
        or (token_count is not None and shape[0] != token_count)
    ):
        raise SyntheticParityError("routed experts have an invalid token/route shape")
    if route.get("dtype") != "uint8" or route.get("start", 0) != 0:
        raise SyntheticParityError("routed experts have invalid dtype or start")
    expected_bytes = math.prod(shape)
    if expected_bytes > MAX_ROUTE_BYTES:
        raise SyntheticParityError("routed experts payload is too large")
    encoded = route.get("data")
    if not isinstance(encoded, str):
        raise SyntheticParityError("routed experts data is missing")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SyntheticParityError("routed experts data is not valid base64") from error
    if len(raw) != expected_bytes:
        raise SyntheticParityError("routed experts data length does not match shape")
    if is_reference_route and (
        route.get("schema_version") != 1
        or route.get("encoding") != "base64"
        or route.get("dense_layers") != DENSE_LAYERS
        or route.get("experts") != EXPERTS
        or route.get("sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise SyntheticParityError("routed experts reference metadata is invalid")
    row_width = shape[1] * shape[2]
    dense_width = DENSE_LAYERS * shape[2]
    if DENSE_LAYERS >= shape[1]:
        raise SyntheticParityError("routed experts have too few layers")
    for offset in range(0, len(raw), row_width):
        row = raw[offset : offset + row_width]
        if any(row[:dense_width]):
            raise SyntheticParityError("dense-layer router IDs must be zero")
        for layer in range(DENSE_LAYERS, shape[1]):
            selected = row[layer * shape[2] : (layer + 1) * shape[2]]
            if any(expert >= EXPERTS for expert in selected):
                raise SyntheticParityError("sparse router IDs are out of range")
            if len(selected) > 1 and len(set(selected)) != len(selected):
                raise SyntheticParityError("sparse router IDs are not distinct")
    normalised = {
        "schema_version": 1,
        "encoding": "base64",
        "dtype": "uint8",
        "shape": list(shape),
        "dense_layers": DENSE_LAYERS,
        "experts": EXPERTS,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": encoded,
    }
    return normalised, raw


def _route_from_raw(raw: bytes, shape_tail: Sequence[int]) -> dict[str, Any]:
    shape = [len(raw) // (int(shape_tail[0]) * int(shape_tail[1])), int(shape_tail[0]), int(shape_tail[1])]
    return {
        "schema_version": 1,
        "encoding": "base64",
        "dtype": "uint8",
        "shape": shape,
        "dense_layers": DENSE_LAYERS,
        "experts": EXPERTS,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(raw).decode(),
    }


def _parse_response(response: Mapping[str, Any], input_ids: Sequence[int]) -> tuple[list[float], dict[str, Any], bytes]:
    if not isinstance(response, Mapping):
        raise SyntheticParityError("H200 response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise SyntheticParityError("H200 response must contain one choice")
    choice = choices[0]
    # The H200 serialized endpoint places prompt_logprobs at response level.
    # Keep this strict so a mock choice-level field cannot hide a schema error.
    if "prompt_logprobs" not in response:
        raise SyntheticParityError("H200 response is missing top-level prompt logprobs")
    logprobs = _flatten_prompt_logprobs(response["prompt_logprobs"], input_ids)
    route, raw = _route_raw(choice.get("routed_experts"), len(input_ids))
    return logprobs, route, raw


def _hash_packed(packed: Mapping[str, Any]) -> dict[str, str]:
    return {
        "input_ids": json_sha256(packed["input_ids"]),
        "inference_logprobs": json_sha256(packed["inference_logprobs"]),
        "position_ids": json_sha256(packed["position_ids"]),
        "loss_mask": json_sha256(packed["loss_mask"]),
        "sequence_lengths": json_sha256(packed["sequence_lengths"]),
        "routed_experts": packed["routed_experts"]["sha256"],
        "reference": json_sha256(
            {
                "input_ids": packed["input_ids"],
                "inference_logprobs": packed["inference_logprobs"],
                "loss_mask": packed["loss_mask"],
                "sequence_lengths": packed["sequence_lengths"],
                "routed_experts": packed["routed_experts"],
            }
        ),
    }


def _build_individual(
    records: Sequence[Mapping[str, Any]],
    logprobs: Sequence[Sequence[float]],
    routes: Sequence[Mapping[str, Any]],
    route_raw: Sequence[bytes],
) -> dict[str, Any]:
    max_length = max(int(record["length"]) for record in records)
    first_route = routes[0]
    tail = first_route["shape"][1:]
    width = math.prod(tail)
    microbatches = []
    for index, (record, values, route, raw) in enumerate(
        zip(records, logprobs, routes, route_raw)
    ):
        length = int(record["length"])
        pad_count = max_length - length
        # Repeat one valid route row for padding.  The trainer can therefore
        # consume every row while loss and route masks exclude padding.
        last_row = raw[-width:]
        padded_raw = raw + last_row * pad_count
        padded_route = _route_from_raw(padded_raw, tail)
        mask = [False] + [True] * (length - 1) + [False] * pad_count
        microbatches.append(
            {
                "microbatch_index": index,
                "sequence_index": index,
                "sequence_length": length,
                "original_sequence_length": length,
                "padding_count": pad_count,
                "input_ids": list(record["input_ids"]) + [PADDING_TOKEN_ID] * pad_count,
                "inference_logprobs": list(values) + [0.0] * pad_count,
                "position_ids": list(range(max_length)),
                "loss_mask": mask,
                "padding_mask": [not flag for flag in mask],
                "route_mask": mask,
                "sequence_lengths": [max_length],
                "routed_experts": padded_route,
                "source_kind": SOURCE_KIND,
            }
        )
    return {"max_seq_len": max_length, "microbatches": microbatches}


def _manifest(
    records: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    packed: Mapping[str, Any],
    individual: Mapping[str, Any],
    base_model: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "diagnostic_name": DIAGNOSTIC_NAME,
        "source_kind": SOURCE_KIND,
        "base_model": base_model,
        "topology": dict(TOPOLOGY),
        "sequence_count": len(records),
        "total_tokens": sum(int(item["length"]) for item in records),
        "sequence_lengths": [int(item["length"]) for item in records],
        "token_count": TOTAL_TOKENS,
        "sequence_sha256": [str(item["sha256"]) for item in records],
        "request_sha256": [json_sha256(item) for item in requests],
        "response_sha256": [json_sha256(item) for item in responses],
        "route_shapes": [list(route["shape"]) for route in routes],
        "route_sha256": [str(route["sha256"]) for route in routes],
        "route_construction": ROUTE_CONSTRUCTION,
        "pp_transfer_sparse_indices": PP_TRANSFER_SPARSE_INDICES,
        "packed_hashes": _hash_packed(packed),
        "individual_hashes": [
            json_sha256(item)
            for item in individual["microbatches"]
        ],
        "parity_gates": dict(PARITY_GATES),
    }


def _packed_reference(packed: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one direct-parity reference accepted by FakeDataLoader."""
    return {
        "schema_version": 1,
        "diagnostic_name": DIAGNOSTIC_NAME,
        "source_kind": SOURCE_KIND,
        "input_ids": list(packed["input_ids"]),
        "inference_logprobs": list(packed["inference_logprobs"]),
        "position_ids": list(packed["position_ids"]),
        "loss_mask": list(packed["loss_mask"]),
        "sequence_lengths": list(packed["sequence_lengths"]),
        "routed_experts": dict(packed["routed_experts"]),
    }


def _individual_reference(
    individual: Mapping[str, Any],
    base_model: str,
) -> dict[str, Any]:
    """Return schema-v2 all-microbatch input for five trainer steps."""
    return {
        "schema_version": 2,
        "diagnostic_name": DIAGNOSTIC_NAME,
        "source_kind": SOURCE_KIND,
        "base_model": base_model,
        "sequence_lengths": list(SEQUENCE_LENGTHS),
        "token_count": TOTAL_TOKENS,
        "microbatches": [dict(item) for item in individual["microbatches"]],
    }


def _endpoint_url(endpoint: str) -> str:
    """Require the exact serialized H200 generate endpoint path."""
    if not isinstance(endpoint, str) or not endpoint:
        raise SyntheticParityError("H200 generate endpoint is required")
    value = endpoint.rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path != "/inference/v1/generate"
        or parsed.query
        or parsed.fragment
    ):
        raise SyntheticParityError(
            "endpoint must be the serialized /inference/v1/generate URL"
        )
    return value


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "manifest.json",
        "packed": output_dir / "packed-reference.json",
        "individual": output_dir / "individual-reference.json",
    }
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise SyntheticParityError("synthetic parity output already exists; refusing to clobber")
    return paths


def _write_json_bundle_no_clobber(
    payloads: Mapping[Path, Mapping[str, Any]],
) -> None:
    """Install a complete bundle with exclusive, same-directory links."""
    temporary: dict[Path, Path] = {}
    linked: list[Path] = []
    try:
        for destination, payload in payloads.items():
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.partial-",
                delete=False,
            )
            temporary_path = Path(handle.name)
            temporary[destination] = temporary_path
            with handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        for destination, temporary_path in temporary.items():
            try:
                os.link(temporary_path, destination)
            except FileExistsError as error:
                raise SyntheticParityError(
                    "synthetic parity output appeared during install; refusing to clobber"
                ) from error
            linked.append(destination)
    except Exception:
        for destination in linked:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary_path in temporary.values():
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def build_artifact(
    *,
    endpoint: str,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    seed: str = DEFAULT_SEED,
    sequences: Sequence[Any] | None = None,
    timeout: float = 120.0,
    post_json: Callable[[str, Mapping[str, Any], float], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Request and write one deterministic packed/individual artifact bundle."""
    endpoint = _endpoint_url(endpoint)
    if output_dir is None:
        if output_path is None:
            raise SyntheticParityError("synthetic parity output_dir is required")
        output_dir = Path(output_path).parent
    paths = _artifact_paths(Path(output_dir))
    if sequences is None:
        records = make_synthetic_sequences(seed)
    else:
        records = _normalise_sequences(sequences)
    requests = make_request_descriptions(records, base_model)
    if post_json is None:
        post_json = _post_json
    responses: list[Mapping[str, Any]] = []
    logs: list[list[float]] = []
    routes: list[dict[str, Any]] = []
    route_raw: list[bytes] = []
    for request, record in zip(requests, records):
        response = post_json(endpoint, request, timeout)
        if not isinstance(response, Mapping):
            raise SyntheticParityError("H200 response must be an object")
        values, route, raw = _parse_response(response, record["input_ids"])
        responses.append(dict(response))
        logs.append(values)
        routes.append(route)
        route_raw.append(raw)
    route_tail = routes[0]["shape"][1:]
    if any(route["shape"][1:] != route_tail for route in routes):
        raise SyntheticParityError("individual routed expert shapes do not match")
    all_ids = [token for record in records for token in record["input_ids"]]
    all_logs = [value for values in logs for value in values]
    all_raw = b"".join(route_raw)
    packed_mask = []
    for length in (int(record["length"]) for record in records):
        packed_mask.extend([False] + [True] * (length - 1))
    packed = {
        "source_kind": SOURCE_KIND,
        "input_ids": all_ids,
        "inference_logprobs": all_logs,
        "position_ids": [
            position
            for length in (int(record["length"]) for record in records)
            for position in range(length)
        ],
        "loss_mask": packed_mask,
        "padding_mask": [False] * len(all_ids),
        "route_mask": [True] * len(all_ids),
        "sequence_lengths": [int(record["length"]) for record in records],
        "routed_experts": _route_from_raw(all_raw, route_tail),
    }
    individual = _build_individual(records, logs, routes, route_raw)
    packed_reference = _packed_reference(packed)
    individual_reference = _individual_reference(individual, base_model)
    manifest = _manifest(
        records, requests, responses, routes, packed, individual, base_model
    )
    artifact = {
        "schema_version": 1,
        "artifact_kind": "trinity_synthetic_packed_individual_parity",
        "source_kind": SOURCE_KIND,
        "base_model": base_model,
        "topology": dict(TOPOLOGY),
        "sequence_lengths": [int(record["length"]) for record in records],
        "sequence_count": len(records),
        "total_tokens": len(all_ids),
        "sequences": [
            {"index": index, "length": record["length"], "sha256": record["sha256"]}
            for index, record in enumerate(records)
        ],
        "requests": [dict(request) for request in requests],
        "responses": responses,
        "packed": packed,
        "individual": individual,
    }
    artifact["manifest"] = manifest
    artifact["packed_reference"] = packed_reference
    artifact["individual_reference"] = individual_reference
    validate_artifact(artifact)
    _write_json_bundle_no_clobber(
        {
            paths["manifest"]: manifest,
            paths["packed"]: packed_reference,
            paths["individual"]: individual_reference,
        }
    )
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all bindings and reject token, route, or response tampering."""
    if not isinstance(value, Mapping):
        raise SyntheticParityError("artifact must be an object")
    if value.get("schema_version") != 1 or value.get("artifact_kind") != "trinity_synthetic_packed_individual_parity":
        raise SyntheticParityError("artifact schema is not synthetic packed parity")
    validate_source_kind(value)
    if value.get("base_model") != value.get("manifest", {}).get("base_model"):
        raise SyntheticParityError("manifest base model binding differs")
    if value.get("topology") != TOPOLOGY:
        raise SyntheticParityError("artifact topology differs from CP8/PP8/EP8")
    records = value.get("sequences")
    requests = value.get("requests")
    responses = value.get("responses")
    manifest = value.get("manifest")
    packed = value.get("packed")
    individual = value.get("individual")
    if not all(isinstance(item, list) for item in (records, requests, responses)) or not isinstance(manifest, Mapping) or not isinstance(packed, Mapping) or not isinstance(individual, Mapping):
        raise SyntheticParityError("artifact sections are incomplete")
    if len(records) != SEQUENCE_COUNT or len(requests) != SEQUENCE_COUNT or len(responses) != SEQUENCE_COUNT:
        raise SyntheticParityError("artifact must contain five requests and responses")
    lengths = list(SEQUENCE_LENGTHS)
    if value.get("sequence_lengths") != lengths or manifest.get("sequence_lengths") != lengths:
        raise SyntheticParityError("artifact sequence lengths differ")
    if value.get("sequence_count") != SEQUENCE_COUNT or value.get("total_tokens") != TOTAL_TOKENS:
        raise SyntheticParityError("artifact sequence counts differ")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("diagnostic_name") != DIAGNOSTIC_NAME
        or manifest.get("token_count") != TOTAL_TOKENS
        or manifest.get("route_construction") != ROUTE_CONSTRUCTION
        or manifest.get("pp_transfer_sparse_indices") is not PP_TRANSFER_SPARSE_INDICES
    ):
        raise SyntheticParityError("manifest diagnostic binding differs")
    normalised_records = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("index") != index or record.get("length") != lengths[index]:
            raise SyntheticParityError("sequence metadata is not bound")
        # Reconstructing token IDs is not required.  The packed hash and the
        # sequence digest bind exact IDs supplied to each request.
        request = requests[index]
        if not isinstance(request, Mapping) or request.get("token_ids") is None:
            raise SyntheticParityError("request token IDs are missing")
        tokens = request["token_ids"]
        if not isinstance(tokens, list) or len(tokens) != lengths[index] or json_sha256(tokens) != record.get("sha256"):
            raise SyntheticParityError("sequence token hash differs")
        normalised_records.append({"input_ids": tokens, "length": lengths[index], "sha256": record.get("sha256")})
    expected_requests = make_request_descriptions(normalised_records, value["base_model"])
    if list(requests) != expected_requests:
        raise SyntheticParityError("request description differs from exact base-model parameters")
    if manifest.get("request_sha256") != [json_sha256(item) for item in requests]:
        raise SyntheticParityError("request hash binding differs")
    parsed_logs: list[list[float]] = []
    parsed_routes: list[dict[str, Any]] = []
    parsed_raw: list[bytes] = []
    for index, response in enumerate(responses):
        logs, route, raw = _parse_response(response, normalised_records[index]["input_ids"])
        parsed_logs.append(logs)
        parsed_routes.append(route)
        parsed_raw.append(raw)
    if manifest.get("response_sha256") != [json_sha256(item) for item in responses]:
        raise SyntheticParityError("response hash binding differs")
    if manifest.get("route_shapes") != [route["shape"] for route in parsed_routes] or manifest.get("route_sha256") != [route["sha256"] for route in parsed_routes]:
        raise SyntheticParityError("route shape or hash binding differs")
    all_ids = [token for record in normalised_records for token in record["input_ids"]]
    all_logs = [value for logs in parsed_logs for value in logs]
    all_raw = b"".join(parsed_raw)
    expected_packed_mask = []
    for length in lengths:
        expected_packed_mask.extend([False] + [True] * (length - 1))
    expected_positions = [
        position for length in lengths for position in range(length)
    ]
    if (
        packed.get("input_ids") != all_ids
        or packed.get("inference_logprobs") != all_logs
        or packed.get("position_ids") != expected_positions
        or packed.get("sequence_lengths") != lengths
    ):
        raise SyntheticParityError("packed token or logprob concatenation differs")
    if packed.get("loss_mask") != expected_packed_mask or packed.get("padding_mask") != [False] * TOTAL_TOKENS or packed.get("route_mask") != [True] * TOTAL_TOKENS:
        raise SyntheticParityError("packed masks are not exact")
    try:
        packed_route, packed_bytes = _route_raw(
            packed.get("routed_experts"), TOTAL_TOKENS
        )
    except SyntheticParityError as error:
        raise SyntheticParityError(f"packed route is invalid: {error}") from error
    if packed_bytes != all_raw or packed_route["shape"][1:] != parsed_routes[0]["shape"][1:]:
        raise SyntheticParityError("packed route concatenation differs")
    if manifest.get("packed_hashes") != _hash_packed(packed):
        raise SyntheticParityError("packed hash binding differs")
    microbatches = individual.get("microbatches")
    max_length = individual.get("max_seq_len")
    if max_length != max(lengths) or not isinstance(microbatches, list) or len(microbatches) != SEQUENCE_COUNT:
        raise SyntheticParityError("individual microbatches are incomplete")
    expected_individual_hashes = []
    width = math.prod(parsed_routes[0]["shape"][1:])
    for index, microbatch in enumerate(microbatches):
        if not isinstance(microbatch, Mapping):
            raise SyntheticParityError("individual microbatch is invalid")
        length = lengths[index]
        pad_count = max_length - length
        expected_ids = list(normalised_records[index]["input_ids"]) + [PADDING_TOKEN_ID] * pad_count
        expected_mask = [False] + [True] * (length - 1) + [False] * pad_count
        if (
            microbatch.get("input_ids") != expected_ids
            or microbatch.get("inference_logprobs") != parsed_logs[index] + [0.0] * pad_count
            or microbatch.get("position_ids") != list(range(max_length))
            or microbatch.get("loss_mask") != expected_mask
            or microbatch.get("padding_mask") != [not flag for flag in expected_mask]
            or microbatch.get("route_mask") != expected_mask
            or microbatch.get("sequence_lengths") != [max_length]
            or microbatch.get("sequence_length") != length
        ):
            raise SyntheticParityError("individual padding or mask binding differs")
        route, raw = _route_raw(microbatch.get("routed_experts"), max_length)
        expected_raw = parsed_raw[index] + parsed_raw[index][-width:] * pad_count
        if raw != expected_raw or route["shape"][1:] != parsed_routes[index]["shape"][1:]:
            raise SyntheticParityError("individual route rows differ")
        expected_individual_hashes.append(json_sha256(microbatch))
    if manifest.get("individual_hashes") != expected_individual_hashes:
        raise SyntheticParityError("individual hash binding differs")
    if manifest.get("parity_gates") != PARITY_GATES:
        raise SyntheticParityError("parity gates changed")
    packed_reference = value.get("packed_reference")
    individual_reference = value.get("individual_reference")
    if packed_reference != _packed_reference(packed):
        raise SyntheticParityError("packed FakeDataLoader reference differs")
    if individual_reference != _individual_reference(individual, value["base_model"]):
        raise SyntheticParityError("individual FakeDataLoader reference differs")
    return dict(value)


def _load_mapping(value: Path | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, Path):
        try:
            value = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SyntheticParityError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        raise SyntheticParityError(f"{label} must be an object")
    return dict(value)


def _as_reference(value: Path | Mapping[str, Any], label: str) -> dict[str, Any]:
    value = _load_mapping(value, label)
    if label == "packed reference" and isinstance(value.get("packed_reference"), Mapping):
        value = dict(value["packed_reference"])
    elif label == "packed reference" and isinstance(value.get("packed"), Mapping):
        value = dict(value["packed"])
    if label == "individual reference" and isinstance(value.get("individual_reference"), Mapping):
        value = dict(value["individual_reference"])
    elif label == "individual reference" and isinstance(value.get("individual"), Mapping):
        value = dict(value["individual"])
    return value


def _numeric_array(value: Any, label: str, count: int) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise SyntheticParityError(f"{label} must contain {count} values")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SyntheticParityError(f"{label} contains a non-numeric value")
        item = float(item)
        if not math.isfinite(item):
            raise SyntheticParityError(f"{label} contains a non-finite value")
        result.append(item)
    return result


def _stats(
    left: Sequence[float],
    right: Sequence[float],
    *,
    include_kl: bool = False,
) -> dict[str, float | int]:
    if len(left) != len(right) or not left:
        raise SyntheticParityError("comparison arrays are not aligned")
    differences = sorted(abs(float(a) - float(b)) for a, b in zip(left, right))
    mismatch_count = sum(delta > 1e-4 for delta in differences)
    p95_index = min(len(differences) - 1, math.ceil(len(differences) * 0.95) - 1)
    result: dict[str, float | int] = {
        "mean_abs": sum(differences) / len(differences),
        "p95_abs": differences[p95_index],
        "max_abs": differences[-1],
        "mismatch_count": mismatch_count,
        "count": len(differences),
    }
    if include_kl:
        kl_values = []
        for actual, reference in zip(left, right):
            delta = float(actual) - float(reference)
            try:
                kl_values.append(math.exp(delta) - delta - 1.0)
            except OverflowError:
                kl_values.append(float("inf"))
        result["mean_kl"] = sum(kl_values) / len(kl_values)
    return result


def compare_packed_individual(
    packed: Path | Mapping[str, Any],
    individual: Path | Mapping[str, Any],
    reference: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare trainer deltas from direct packed and schema-v2 microbatches.

    The function binds token IDs, position IDs, sequence lengths, and masks
    before comparing.  It accepts either the emitted reference files or the
    in-memory aggregate returned by :func:`build_artifact`.
    """
    packed_ref = _as_reference(packed, "packed reference")
    individual_ref = _as_reference(individual, "individual reference")
    reference_ref = (
        _as_reference(reference, "packed reference")
        if reference is not None
        else packed_ref
    )
    if individual_ref.get("schema_version") != 2:
        raise SyntheticParityError("individual reference must use schema version 2")
    packed_ids = packed_ref.get("input_ids")
    packed_lengths = packed_ref.get("sequence_lengths")
    expected_positions = [
        position for length in SEQUENCE_LENGTHS for position in range(length)
    ]
    if "position_ids" not in packed_ref or "loss_mask" not in packed_ref:
        raise SyntheticParityError("packed reference requires position_ids and loss_mask")
    packed_positions = packed_ref["position_ids"]
    packed_mask = packed_ref["loss_mask"]
    expected_mask = []
    for length in SEQUENCE_LENGTHS:
        expected_mask.extend([False] + [True] * (length - 1))
    if (
        not isinstance(packed_ids, list)
        or packed_lengths != list(SEQUENCE_LENGTHS)
        or not isinstance(packed_positions, list)
        or packed_positions != expected_positions
        or not isinstance(packed_mask, list)
        or len(packed_mask) != TOTAL_TOKENS
    ):
        raise SyntheticParityError("packed IDs, positions, sequence lengths, or mask are invalid")
    packed_reference = _numeric_array(
        reference_ref.get(
            "reference_logprobs", reference_ref.get("inference_logprobs")
        ),
        "packed reference logprobs",
        TOTAL_TOKENS,
    )
    if (
        reference_ref.get("input_ids", packed_ids) != packed_ids
        or reference_ref.get("sequence_lengths", packed_lengths) != packed_lengths
    ):
        raise SyntheticParityError("packed and reference token bindings differ")
    packed_trainer = _numeric_array(
        packed_ref.get("trainer_logprobs", packed_reference),
        "packed trainer logprobs",
        TOTAL_TOKENS,
    )
    if packed_mask != expected_mask:
        raise SyntheticParityError("packed loss mask is not sequence aligned")
    microbatches = individual_ref.get("microbatches")
    if not isinstance(microbatches, list) or len(microbatches) != SEQUENCE_COUNT:
        raise SyntheticParityError("individual reference does not contain five microbatches")
    individual_ids: list[int] = []
    individual_reference: list[float] = []
    individual_trainer: list[float] = []
    offsets = []
    offset = 0
    for index, (length, microbatch) in enumerate(zip(SEQUENCE_LENGTHS, microbatches)):
        if not isinstance(microbatch, Mapping):
            raise SyntheticParityError("individual microbatch is invalid")
        padded_ids = microbatch.get("input_ids")
        if not isinstance(padded_ids, list) or len(padded_ids) != max(SEQUENCE_LENGTHS):
            raise SyntheticParityError("individual input IDs are not padded to max length")
        sequence_length = microbatch.get(
            "original_sequence_length",
            microbatch.get("sequence_length", length),
        )
        if sequence_length != length or microbatch.get("sequence_lengths") != [max(SEQUENCE_LENGTHS)]:
            raise SyntheticParityError("individual sequence lengths are not bound")
        for field in ("position_ids", "microbatch_index", "microbatch_count"):
            if field not in microbatch:
                raise SyntheticParityError(
                    f"individual microbatch requires {field}"
                )
        positions = microbatch["position_ids"]
        if positions != list(range(max(SEQUENCE_LENGTHS))):
            raise SyntheticParityError("individual positions are not bound")
        mask = microbatch.get("loss_mask")
        expected_micro_mask = [False] + [True] * (length - 1) + [False] * (max(SEQUENCE_LENGTHS) - length)
        if mask != expected_micro_mask:
            raise SyntheticParityError("individual loss mask is not bound")
        actual_ids = padded_ids[:length]
        if actual_ids != packed_ids[offset : offset + length]:
            raise SyntheticParityError(f"individual input IDs differ at sequence {index}")
        if (
            microbatch.get("microbatch_index", index) != index
            or microbatch.get("microbatch_count", SEQUENCE_COUNT) != SEQUENCE_COUNT
        ):
            raise SyntheticParityError("individual microbatch index/count is not bound")
        ref_values = _numeric_array(
            microbatch.get(
                "reference_logprobs", microbatch.get("inference_logprobs")
            ),
            f"individual reference logprobs {index}",
            max(SEQUENCE_LENGTHS),
        )
        trainer_values = _numeric_array(
            microbatch.get("trainer_logprobs", ref_values),
            f"individual trainer logprobs {index}",
            max(SEQUENCE_LENGTHS),
        )
        individual_ids.extend(actual_ids)
        individual_reference.extend(ref_values[:length])
        individual_trainer.extend(trainer_values[:length])
        offsets.append({"index": index, "start": offset, "end": offset + length, "length": length})
        offset += length
    if individual_ids != packed_ids or individual_reference != packed_reference:
        raise SyntheticParityError("packed and individual reference bindings differ")
    sequence_reports = []
    for item in offsets:
        start, end = item["start"], item["end"]
        valid_start = start + 1
        sequence_reports.append(
            {
                **item,
                "packed_vs_individual": _stats(
                    packed_trainer[valid_start:end], individual_trainer[valid_start:end]
                ),
                "packed_vs_reference": _stats(
                    packed_trainer[valid_start:end],
                    packed_reference[valid_start:end],
                    include_kl=True,
                ),
                "individual_vs_reference": _stats(
                    individual_trainer[valid_start:end],
                    individual_reference[valid_start:end],
                    include_kl=True,
                ),
            }
        )
    packed_valid = [value for value, flag in zip(packed_trainer, packed_mask) if flag]
    packed_reference_valid = [value for value, flag in zip(packed_reference, packed_mask) if flag]
    individual_valid = []
    individual_reference_valid = []
    for index, length in enumerate(SEQUENCE_LENGTHS):
        microbatch = microbatches[index]
        mask = microbatch["loss_mask"]
        trainer_values = _numeric_array(
            microbatch.get("trainer_logprobs", microbatch.get("reference_logprobs")),
            f"individual trainer logprobs {index}",
            max(SEQUENCE_LENGTHS),
        )
        reference_values = _numeric_array(
            microbatch.get("reference_logprobs", microbatch.get("inference_logprobs")),
            f"individual reference logprobs {index}",
            max(SEQUENCE_LENGTHS),
        )
        individual_valid.extend(value for value, flag in zip(trainer_values, mask) if flag)
        individual_reference_valid.extend(value for value, flag in zip(reference_values, mask) if flag)
    return {
        "schema_version": 1,
        "diagnostic_name": DIAGNOSTIC_NAME,
        "sequence_lengths": list(SEQUENCE_LENGTHS),
        "token_count": TOTAL_TOKENS,
        "sequences": sequence_reports,
        "totals": {
            "packed_vs_individual": _stats(packed_valid, individual_valid),
            "packed_vs_reference": _stats(
                packed_valid, packed_reference_valid, include_kl=True
            ),
            "individual_vs_reference": _stats(
                individual_valid, individual_reference_valid, include_kl=True
            ),
        },
    }


def _post_json(endpoint: str, body: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        endpoint,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (OSError, HTTPError, URLError, json.JSONDecodeError) as error:
        raise SyntheticParityError(f"synthetic H200 request failed: {error}") from error
    if not isinstance(value, dict):
        raise SyntheticParityError("H200 response must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--timeout", default=120.0, type=float)
    args = parser.parse_args(argv)
    build_artifact(
        endpoint=args.endpoint,
        output_dir=args.output_dir,
        base_model=args.base_model,
        seed=args.seed,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
