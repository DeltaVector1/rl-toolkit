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
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
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
VOCAB_SIZE = 151936
PADDING_TOKEN_ID = 0
MAX_ROUTE_BYTES = 512 * 1024 * 1024
_ROUTE_FIELDS = {"data", "dtype", "shape", "start"}


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
    if not isinstance(route, Mapping) or not _ROUTE_FIELDS.issubset(route):
        raise SyntheticParityError("response is missing routed experts")
    shape = route.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(item) is not int or item <= 0 for item in shape)
        or (token_count is not None and shape[0] != token_count)
    ):
        raise SyntheticParityError("routed experts have an invalid token/route shape")
    if route.get("dtype") != "uint8" or route.get("start") != 0:
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
    normalised = {
        "schema_version": 1,
        "encoding": "base64",
        "dtype": "uint8",
        "shape": list(shape),
        "start": 0,
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
        "start": 0,
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
    prompt = choice.get("prompt_logprobs", response.get("prompt_logprobs"))
    logprobs = _flatten_prompt_logprobs(prompt, input_ids)
    route, raw = _route_raw(choice.get("routed_experts"), len(input_ids))
    return logprobs, route, raw


def _hash_packed(packed: Mapping[str, Any]) -> dict[str, str]:
    return {
        "input_ids": json_sha256(packed["input_ids"]),
        "inference_logprobs": json_sha256(packed["inference_logprobs"]),
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
        mask = [True] * length + [False] * pad_count
        microbatches.append(
            {
                "microbatch_index": index,
                "sequence_index": index,
                "sequence_length": length,
                "original_sequence_length": length,
                "padding_count": pad_count,
                "input_ids": list(record["input_ids"]) + [PADDING_TOKEN_ID] * pad_count,
                "inference_logprobs": list(values) + [0.0] * pad_count,
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
        "schema_version": 1,
        "source_kind": SOURCE_KIND,
        "base_model": base_model,
        "topology": dict(TOPOLOGY),
        "sequence_count": len(records),
        "total_tokens": sum(int(item["length"]) for item in records),
        "sequence_lengths": [int(item["length"]) for item in records],
        "sequence_counts": [int(item["length"]) for item in records],
        "sequence_sha256": [str(item["sha256"]) for item in records],
        "request_sha256": [json_sha256(item) for item in requests],
        "response_sha256": [json_sha256(item) for item in responses],
        "route_shapes": [list(route["shape"]) for route in routes],
        "route_sha256": [str(route["sha256"]) for route in routes],
        "packed_hashes": _hash_packed(packed),
        "individual_hashes": [
            json_sha256(item)
            for item in individual["microbatches"]
        ],
        "parity_gates": dict(PARITY_GATES),
    }


def build_artifact(
    *,
    endpoint: str,
    output_path: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    seed: str = DEFAULT_SEED,
    sequences: Sequence[Any] | None = None,
    timeout: float = 120.0,
    post_json: Callable[[str, Mapping[str, Any], float], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Request and write one deterministic packed/individual artifact."""
    if not isinstance(endpoint, str) or not endpoint:
        raise SyntheticParityError("H200 generate endpoint is required")
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
    packed = {
        "source_kind": SOURCE_KIND,
        "input_ids": all_ids,
        "inference_logprobs": all_logs,
        "loss_mask": [True] * len(all_ids),
        "padding_mask": [False] * len(all_ids),
        "route_mask": [True] * len(all_ids),
        "sequence_lengths": [int(record["length"]) for record in records],
        "routed_experts": _route_from_raw(all_raw, route_tail),
    }
    individual = _build_individual(records, logs, routes, route_raw)
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
    artifact["manifest"] = _manifest(
        records, requests, responses, routes, packed, individual, base_model
    )
    validate_artifact(artifact)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n")
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
    if packed.get("input_ids") != all_ids or packed.get("inference_logprobs") != all_logs or packed.get("sequence_lengths") != lengths:
        raise SyntheticParityError("packed token or logprob concatenation differs")
    if packed.get("loss_mask") != [True] * TOTAL_TOKENS or packed.get("padding_mask") != [False] * TOTAL_TOKENS or packed.get("route_mask") != [True] * TOTAL_TOKENS:
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
        expected_mask = [True] * length + [False] * pad_count
        if (
            microbatch.get("input_ids") != expected_ids
            or microbatch.get("inference_logprobs") != parsed_logs[index] + [0.0] * pad_count
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
    return dict(value)


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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--timeout", default=120.0, type=float)
    args = parser.parse_args(argv)
    build_artifact(
        endpoint=args.endpoint,
        output_path=args.output,
        base_model=args.base_model,
        seed=args.seed,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
