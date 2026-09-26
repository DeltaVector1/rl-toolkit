"""Build one trainer-only Trinity parity reference from a real prompt.

The input record is deliberately separate from rollout batches.  It must be a
JSON object with ``input_ids`` and ``sequence_lengths``.  An optional
``expected_route_shape`` and ``expected_route_sha256`` make the route check
stricter when those values are already recorded with the prompt.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_RAW_ROUTE_FIELDS = {"data", "dtype", "shape", "start"}
_MAX_ROUTE_BYTES = 512 * 1024 * 1024
_POLICIES = {"base", "policy-0"}
_PREFILL_RESCORE_REQUEST_MODE = "serialized_generate_decode256_prefill_rescore"
_PREFILL_RESCORE_PROMPT_TOKENS = 256
_PREFILL_RESCORE_COMPLETION_TOKENS = 256
_PREFILL_RESCORE_SEQUENCE_TOKENS = 512
_DEFAULT_TOKEN_COUNT = 8160
# A two-percent excess limit rejects a periodic seed while allowing ordinary
# syntax to recur in long natural-language input.
_REPETITION_WINDOW = 16
_MAX_REPEATED_WINDOW_FRACTION = 0.02
_NON_ROLLOUT_SOURCES = {
    "deterministic_local_tokenizer_seed",
    "hub_cache_non_rollout",
    "operator_non_rollout",
}
_POLICY_RECEIPT_DIGEST_FIELDS = ("target_sha256", "config_sha256")
_POLICY_RECEIPT_MATCH_FIELDS = (
    "target_sha256",
    "target_size",
    "config_sha256",
    "config_size",
    "source_url",
    "lora_name",
    "server_instance",
)


class ReferenceError(ValueError):
    """The preserved prompt or serialized generate response is unsafe to use."""


_SEED_TEMPLATES = (
    "Ledger sample {index}: the {color} {subject} {verb} {object}; record "
    "window {window}, checksum {digest}, and policy note {note}.",
    "For case {index}, inspect {subject} near the {color} {object}; preserve "
    "the {verb} route, window {window}, digest {digest}, and note {note}.",
    "Route memo {index} says the {object} is {verb} by a {color} {subject}; "
    "retain window {window}, digest {digest}, and policy note {note}.",
    "At ledger position {index}, compare the {color} {subject} with {object}; "
    "the selected action is {verb}, window {window}, digest {digest}, note {note}.",
)
_SEED_COLORS = ("amber", "cobalt", "coral", "emerald", "indigo", "silver", "violet")
_SEED_SUBJECTS = ("beacon", "cartographer", "compiler", "curator", "navigator", "scribe", "surveyor")
_SEED_VERBS = ("balances", "checks", "maps", "measures", "revises", "stabilizes", "verifies")
_SEED_OBJECTS = ("a quiet harbor", "the cedar archive", "a copper relay", "the lunar index", "a paper atlas", "the river switch", "a winter garden")


def _seed_text(blocks: int) -> str:
    """Return deterministic prose with indexed and hashed local variation."""
    lines = []
    for index in range(blocks):
        digest = hashlib.sha256(f"trinity-qualifier-seed:{index}".encode()).hexdigest()[:12]
        values = [
            index,
            _SEED_COLORS[int(digest[0:2], 16) % len(_SEED_COLORS)],
            _SEED_SUBJECTS[int(digest[2:4], 16) % len(_SEED_SUBJECTS)],
            _SEED_VERBS[int(digest[4:6], 16) % len(_SEED_VERBS)],
            _SEED_OBJECTS[int(digest[6:8], 16) % len(_SEED_OBJECTS)],
            f"{int(digest[8:10], 16)}-{index % 997}",
            digest,
            f"{(index * 37) % 1009}",
        ]
        lines.append(_SEED_TEMPLATES[index % len(_SEED_TEMPLATES)].format(
            index=values[0],
            color=values[1],
            subject=values[2],
            verb=values[3],
            object=values[4],
            window=values[5],
            digest=values[6],
            note=values[7],
        ))
    return "\n".join(lines)


def _validate_token_diversity(input_ids: list[int]) -> None:
    if len(set(input_ids)) < 64:
        raise ReferenceError("deterministic seed did not produce diverse token IDs")
    if len(input_ids) < _REPETITION_WINDOW:
        return
    windows = [
        tuple(input_ids[index : index + _REPETITION_WINDOW])
        for index in range(len(input_ids) - _REPETITION_WINDOW + 1)
    ]
    repeated_excess = sum(count - 1 for count in Counter(windows).values() if count > 1)
    if repeated_excess > int(len(windows) * _MAX_REPEATED_WINDOW_FRACTION):
        raise ReferenceError(
            "deterministic seed has excessive repeated 16-token windows"
        )


def write_input_record(
    output_path: Path,
    tokenizer_path: str,
    *,
    token_count: int = _DEFAULT_TOKEN_COUNT,
    prefill_rescore: bool = False,
    encode: Callable[[str], list[int]] | None = None,
) -> dict[str, Any]:
    """Tokenize a fresh deterministic seed locally and write one sequence.

    ``local_files_only`` is used for the production tokenizer load. This mode
    never reads rollout files and never sends model data over SSH.
    """
    if not isinstance(prefill_rescore, bool):
        raise ReferenceError("prefill_rescore must be a boolean")
    minimum_token_count = _PREFILL_RESCORE_PROMPT_TOKENS if prefill_rescore else 4097
    if isinstance(token_count, bool) or token_count < minimum_token_count:
        raise ReferenceError(
            f"input token_count must be at least {minimum_token_count}"
        )
    if encode is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as error:
            raise ReferenceError("cannot load the local tokenizer") from error

        def encode(text: str) -> list[int]:
            value = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not isinstance(value, list) or any(
                isinstance(token, bool) or not isinstance(token, int) for token in value
            ):
                raise ReferenceError("tokenizer returned an invalid input ID list")
            return value

    input_ids: list[int] = []
    for blocks in (256, 512, 1024, 2048, 4096):
        input_ids = encode(_seed_text(blocks))
        if len(input_ids) >= token_count:
            break
    if len(input_ids) < token_count:
        raise ReferenceError("deterministic seed did not produce enough tokens")
    input_ids = input_ids[:token_count]
    _validate_token_diversity(input_ids)
    result = {
        "source_kind": "deterministic_local_tokenizer_seed",
        "tokenizer": tokenizer_path,
        "input_ids": input_ids,
        "sequence_lengths": [token_count],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    return result


def _load_record(record: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, Path):
        try:
            value = json.loads(record.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ReferenceError(f"cannot read input record {record}") from error
    else:
        value = dict(record)
    if not isinstance(value, dict):
        raise ReferenceError("input record must be a JSON object")
    source_kind = value.get("source_kind")
    if source_kind not in _NON_ROLLOUT_SOURCES:
        raise ReferenceError(
            "input record must declare a supported non-rollout source_kind"
        )
    input_ids = value.get("input_ids")
    sequence_lengths = value.get("sequence_lengths")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(isinstance(token, bool) or not isinstance(token, int) for token in input_ids)
    ):
        raise ReferenceError("input record input_ids must be a non-empty integer list")
    if (
        not isinstance(sequence_lengths, list)
        or not sequence_lengths
        or any(
            isinstance(length, bool) or not isinstance(length, int) or length <= 0
            for length in sequence_lengths
        )
        or sum(sequence_lengths) != len(input_ids)
    ):
        raise ReferenceError("input record sequence_lengths do not match input_ids")
    return value


def _normalise_route(route: Mapping[str, Any], token_count: int) -> dict[str, Any]:
    if set(route) != _RAW_ROUTE_FIELDS:
        raise ReferenceError("serialized routed experts have an invalid schema")
    shape = route.get("shape")
    if (
        route.get("dtype") != "uint8"
        or route.get("start") != 0
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        or shape[0] != token_count
    ):
        raise ReferenceError("serialized routed experts have invalid metadata or token count")
    dense_layers = 6
    experts = 256
    if dense_layers >= shape[1]:
        raise ReferenceError("serialized routed experts have too few layers")
    expected_bytes = shape[0] * shape[1] * shape[2]
    encoded = route.get("data")
    if (
        expected_bytes > _MAX_ROUTE_BYTES
        or not isinstance(encoded, str)
        or len(encoded) != 4 * ((expected_bytes + 2) // 3)
    ):
        raise ReferenceError("serialized routed experts have invalid size metadata")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ReferenceError("routed experts data is not valid base64") from error
    if len(raw) != expected_bytes:
        raise ReferenceError("serialized routed experts data has the wrong length")

    row_width = shape[1] * shape[2]
    for offset in range(0, len(raw), row_width):
        row = raw[offset : offset + row_width]
        dense_end = dense_layers * shape[2]
        if any(row[:dense_end]):
            raise ReferenceError("dense-layer router IDs must be zero")
        for layer in range(dense_layers, shape[1]):
            start = (layer * shape[2])
            selected = row[start : start + shape[2]]
            if any(expert >= experts for expert in selected) or len(set(selected)) != len(selected):
                raise ReferenceError("sparse router IDs are invalid")
    return {
        "schema_version": 1,
        "encoding": "base64",
        "dtype": "uint8",
        "shape": shape,
        "dense_layers": dense_layers,
        "experts": experts,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": encoded,
    }


def _flatten_prompt_logprobs(raw: Any, input_ids: list[int]) -> list[float]:
    token_count = len(input_ids)
    if not isinstance(raw, list) or len(raw) != token_count:
        raise ReferenceError("prompt logprob count does not match input token count")
    result: list[float] = []
    for index, entry in enumerate(raw):
        if index == 0 and entry is None:
            result.append(0.0)
            continue
        if not isinstance(entry, dict) or not entry:
            raise ReferenceError(f"prompt logprob entry {index} is missing")
        candidates = [
            value
            for key, value in entry.items()
            if key == input_ids[index] or str(key) == str(input_ids[index])
        ]
        if len(candidates) != 1:
            raise ReferenceError(
                f"prompt logprob entry {index} has no unique value for token {input_ids[index]}"
            )
        candidate = candidates[0]
        if not isinstance(candidate, dict) or isinstance(candidate.get("logprob"), bool):
            raise ReferenceError(f"prompt logprob entry {index} is malformed")
        value = candidate.get("logprob")
        if not isinstance(value, (int, float)):
            raise ReferenceError(f"prompt logprob entry {index} is not numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ReferenceError(f"prompt logprob entry {index} is not finite")
        result.append(value)
    return result


def _flatten_completion_logprobs(raw: Any, completion_ids: list[int]) -> list[float]:
    """Validate the exact generated-token logprob contract."""
    if not isinstance(raw, Mapping):
        raise ReferenceError("completion logprobs must be an object")
    content = raw.get("content")
    if not isinstance(content, list) or len(content) != len(completion_ids):
        raise ReferenceError("completion logprob count does not match generated token count")
    result: list[float] = []
    for index, token_id in enumerate(completion_ids):
        entry = content[index]
        if not isinstance(entry, Mapping):
            raise ReferenceError(f"completion logprob entry {index} is malformed")
        expected_token = f"token_id:{token_id}"
        if entry.get("token") != expected_token:
            raise ReferenceError(
                f"completion logprob entry {index} has the wrong token key"
            )
        value = entry.get("logprob")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReferenceError(f"completion logprob entry {index} is not numeric")
        value = float(value)
        if not math.isfinite(value) or value == -9999.0:
            raise ReferenceError(f"completion logprob entry {index} has no sampling evidence")
        result.append(value)
    return result


def _json_sha256(value: Any) -> str:
    """Hash one response or request without depending on object key order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _route_prediction_comparison(
    decode_route: Mapping[str, Any],
    rescore_route: Mapping[str, Any],
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Compare the route rows that predict the generated suffix."""
    try:
        decode_raw = base64.b64decode(decode_route["data"], validate=True)
        rescore_raw = base64.b64decode(rescore_route["data"], validate=True)
    except (KeyError, binascii.Error, ValueError) as error:
        raise ReferenceError("cannot compare serialized routed experts") from error
    decode_shape = decode_route.get("shape")
    rescore_shape = rescore_route.get("shape")
    if (
        not isinstance(decode_shape, list)
        or not isinstance(rescore_shape, list)
        or len(decode_shape) != 3
        or len(rescore_shape) != 3
        or decode_shape[1:] != rescore_shape[1:]
    ):
        raise ReferenceError("decode and prefill routes have different row schemas")
    decode_width = decode_shape[1] * decode_shape[2]
    rescore_width = rescore_shape[1] * rescore_shape[2]
    if decode_width != rescore_width or len(decode_raw) != decode_shape[0] * decode_width:
        raise ReferenceError("decode route rows are not aligned")
    if len(rescore_raw) != rescore_shape[0] * rescore_width:
        raise ReferenceError("prefill route rows are not aligned")
    # The row immediately before each completion token predicts that token.
    first_row = prompt_tokens - 1
    last_row = first_row + completion_tokens
    if decode_shape[0] < last_row or rescore_shape[0] < last_row:
        raise ReferenceError("decode and prefill routes do not cover the completion")
    differences = []
    for row in range(first_row, last_row):
        decode_ids = list(decode_raw[row * decode_width : (row + 1) * decode_width])
        rescore_ids = list(rescore_raw[row * rescore_width : (row + 1) * rescore_width])
        if decode_ids != rescore_ids:
            differences.append({
                "row": row,
                "decode": decode_ids,
                "prefill": rescore_ids,
            })
    return {
        "first_predicting_row": first_row,
        "last_predicting_row_exclusive": last_row,
        "compared_rows": completion_tokens,
        "different_rows": differences,
        "different_row_count": len(differences),
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
        raise ReferenceError(f"serialized generate request failed: {error}") from error
    if not isinstance(value, dict):
        raise ReferenceError("serialized generate response must be an object")
    return value


def _get_json(endpoint: str, timeout: float) -> dict[str, Any]:
    request = Request(endpoint, headers={"Connection": "close"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (OSError, HTTPError, URLError, json.JSONDecodeError) as error:
        raise ReferenceError(f"staging readiness request failed: {error}") from error
    if not isinstance(value, dict):
        raise ReferenceError("staging readiness response must be an object")
    return value


def _endpoint_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/inference/v1/generate"):
        raise ReferenceError("endpoint must be the serialized /inference/v1/generate URL")
    return endpoint


def _readiness_url(endpoint: str | None, generate_endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ReferenceError("policy-0 requires an explicit staging readiness endpoint")
    value = endpoint.rstrip("/")
    parsed = urlsplit(value)
    generated = urlsplit(generate_endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.path
        or parsed.netloc != generated.netloc
        or value == generate_endpoint
    ):
        raise ReferenceError("staging readiness endpoint must match the active generate endpoint")
    return value


def _load_mapping(value: Path | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, Path):
        try:
            value = json.loads(value.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ReferenceError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        raise ReferenceError(f"{label} must be a JSON object")
    return dict(value)


def _validate_policy_receipt(
    receipt: Path | Mapping[str, Any],
    expected: Path | Mapping[str, Any],
    policy_model: str,
) -> dict[str, Any]:
    actual = _load_mapping(receipt, "policy readiness receipt")
    expected_identity = _load_mapping(expected, "expected policy identity")
    for field in _POLICY_RECEIPT_DIGEST_FIELDS:
        value = actual.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ReferenceError(f"policy receipt has an invalid {field}")
        expected_value = expected_identity.get(field)
        if (
            not isinstance(expected_value, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_value) is None
        ):
            raise ReferenceError(f"expected policy identity has an invalid {field}")
    for field in ("target_size", "config_size"):
        value = actual.get(field)
        if type(value) is not int or value <= (0 if field == "target_size" else -1):
            raise ReferenceError(f"policy receipt has an invalid {field}")
        expected_value = expected_identity.get(field)
        if type(expected_value) is not int or expected_value <= (0 if field == "target_size" else -1):
            raise ReferenceError(f"expected policy identity has an invalid {field}")
        if expected_value != value:
            raise ReferenceError(f"policy receipt differs in {field}")
    source_url = actual.get("source_url")
    parsed_source_url = urlsplit(source_url) if isinstance(source_url, str) else None
    if (
        parsed_source_url is None
        or parsed_source_url.scheme not in {"http", "https"}
        or not parsed_source_url.netloc
    ):
        raise ReferenceError("policy receipt has no source URL")
    lora_name = actual.get("lora_name")
    if lora_name != policy_model:
        raise ReferenceError("policy receipt has the wrong LoRA name")
    server_instance = actual.get("server_instance")
    if not isinstance(server_instance, str) or not server_instance:
        raise ReferenceError("policy receipt has no server instance")
    expected_source = expected_identity.get("source_url")
    expected_source_parts = urlsplit(expected_source) if isinstance(expected_source, str) else None
    if (
        expected_source_parts is None
        or expected_source_parts.scheme not in {"http", "https"}
        or not expected_source_parts.netloc
    ):
        raise ReferenceError("expected policy identity has no source URL")
    for field in _POLICY_RECEIPT_MATCH_FIELDS:
        if expected_identity.get(field) != actual.get(field):
            raise ReferenceError(f"policy receipt differs in {field}")
    return actual


def _extract_policy_receipt(payload: Mapping[str, Any], policy_model: str) -> dict[str, Any]:
    """Extract one active loaded receipt, refusing ambiguous status payloads."""
    if all(field in payload for field in _POLICY_RECEIPT_MATCH_FIELDS):
        return dict(payload)
    direct = payload.get("loaded_receipt")
    if isinstance(direct, Mapping):
        return dict(direct)
    loaded = payload.get("loaded_receipts")
    candidates: list[Any]
    if isinstance(loaded, Mapping):
        candidates = [loaded.get(policy_model)]
    elif isinstance(loaded, list):
        candidates = loaded
    else:
        candidates = []
    candidates = [item for item in candidates if isinstance(item, Mapping)]
    matching = [item for item in candidates if item.get("lora_name") == policy_model]
    if len(matching) != 1:
        raise ReferenceError("staging readiness has no unique loaded policy receipt")
    return dict(matching[0])


def _validate_cache_salt(value: str | None, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ReferenceError("policy-0 requires a fresh cache_salt")
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ReferenceError("cache_salt must be a non-empty string of at most 256 characters")
    return value


def build_reference(
    *,
    endpoint: str,
    input_record: Path | Mapping[str, Any],
    output_path: Path,
    base_model: str,
    policy: str,
    policy_model: str | None = None,
    policy_receipt: Path | Mapping[str, Any] | None = None,
    expected_policy_identity: Path | Mapping[str, Any] | None = None,
    cache_salt: str | None = None,
    prefill_rescore: bool = False,
    rescore_cache_salt: str | None = None,
    readiness_endpoint: str | None = None,
    completion_tokens: int | None = None,
    minimum_completion_tokens: int | None = None,
    trainer_seq_len: int | None = None,
    minimum_tokens: int = 4097,
    expected_route_shape: list[int] | None = None,
    expected_route_sha256: str | None = None,
    timeout: float = 120.0,
    post_json: Callable[[str, Mapping[str, Any], float], dict[str, Any]] = _post_json,
    get_json: Callable[[str, float], dict[str, Any]] = _get_json,
) -> dict[str, Any]:
    if policy not in _POLICIES:
        raise ReferenceError(f"policy must be one of {sorted(_POLICIES)}")
    if not isinstance(prefill_rescore, bool):
        raise ReferenceError("prefill_rescore must be a boolean")
    if not prefill_rescore and rescore_cache_salt is not None:
        raise ReferenceError("rescore_cache_salt requires --prefill-rescore")
    if prefill_rescore and policy != "policy-0":
        raise ReferenceError("prefill-rescore is only supported for policy-0")
    if trainer_seq_len is None:
        trainer_seq_len = (
            _PREFILL_RESCORE_SEQUENCE_TOKENS if prefill_rescore else 8192
        )
    if completion_tokens is None:
        completion_tokens = (
            _PREFILL_RESCORE_COMPLETION_TOKENS if prefill_rescore else 32
        )
    if minimum_completion_tokens is None:
        minimum_completion_tokens = completion_tokens
    if not base_model or (policy == "policy-0" and not policy_model):
        raise ReferenceError("base_model and selected policy model are required")
    if isinstance(minimum_tokens, bool) or minimum_tokens < 0:
        raise ReferenceError("minimum_tokens must be non-negative")
    if (
        isinstance(completion_tokens, bool)
        or completion_tokens < 1
        or completion_tokens > 4096
        or isinstance(minimum_completion_tokens, bool)
        or minimum_completion_tokens < 1
        or minimum_completion_tokens > completion_tokens
        or isinstance(trainer_seq_len, bool)
        or trainer_seq_len < 1
    ):
        raise ReferenceError("completion token limits are invalid")
    record = _load_record(input_record)
    input_ids = list(record["input_ids"])
    token_count = len(input_ids)
    if not prefill_rescore and token_count < minimum_tokens:
        raise ReferenceError(
            f"input sequence has {token_count} tokens; at least {minimum_tokens} are required"
        )
    expected_shape = expected_route_shape or record.get("expected_route_shape")
    if expected_shape is not None and (
        not isinstance(expected_shape, list)
        or len(expected_shape) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in expected_shape)
    ):
        raise ReferenceError("expected_route_shape must be three positive integers")
    expected_digest = expected_route_sha256 or record.get("expected_route_sha256")
    if expected_digest is not None and (
        not isinstance(expected_digest, str) or len(expected_digest) != 64
    ):
        raise ReferenceError("expected_route_sha256 must be a 64-character string")

    model = base_model if policy == "base" else policy_model
    assert model is not None
    if policy == "policy-0":
        if expected_policy_identity is None:
            raise ReferenceError(
                "policy-0 requires a readiness receipt and expected policy identity"
            )
        generate_endpoint = _endpoint_url(endpoint)
        readiness_url = _readiness_url(readiness_endpoint, generate_endpoint)
        expected_identity = _load_mapping(expected_policy_identity, "expected policy identity")
        before_payload = get_json(readiness_url, timeout)
        policy_identity_before = _validate_policy_receipt(
            _extract_policy_receipt(before_payload, model),
            expected_identity,
            model,
        )
        if policy_receipt is not None:
            caller_identity = _validate_policy_receipt(
                policy_receipt,
                expected_identity,
                model,
            )
            if policy_identity_before != caller_identity:
                raise ReferenceError("active readiness receipt differs from caller receipt")
        policy_identity = policy_identity_before
    else:
        if (
            policy_receipt is not None
            or expected_policy_identity is not None
            or readiness_endpoint is not None
        ):
            raise ReferenceError("base policy cannot use a policy readiness receipt")
        policy_identity = None
    cache_salt = _validate_cache_salt(cache_salt, required=policy == "policy-0")
    if policy == "policy-0":
        if prefill_rescore:
            if (
                completion_tokens != _PREFILL_RESCORE_COMPLETION_TOKENS
                or minimum_completion_tokens != _PREFILL_RESCORE_COMPLETION_TOKENS
                or trainer_seq_len != _PREFILL_RESCORE_SEQUENCE_TOKENS
                or token_count != _PREFILL_RESCORE_PROMPT_TOKENS
            ):
                raise ReferenceError(
                    "prefill-rescore qualification requires prompt256, completion256, and seq_len512"
                )
            rescore_cache_salt = _validate_cache_salt(rescore_cache_salt, required=True)
            if cache_salt == rescore_cache_salt:
                raise ReferenceError("decode and prefill cache salts must be distinct")
        elif completion_tokens != 32 or minimum_completion_tokens != 32:
            raise ReferenceError("policy-0 qualification requires exactly 32 generated tokens")
        expected_prompt_tokens = trainer_seq_len - completion_tokens
        if token_count != expected_prompt_tokens:
            raise ReferenceError(
                f"policy-0 prompt must contain exactly {expected_prompt_tokens} tokens"
            )
    response_hashes: dict[str, str] = {}
    request_bodies: dict[str, dict[str, Any]] = {}
    generated_ids: list[int] = []
    generated_logprobs: list[float] = []
    finish_reason: Any = None
    full_ids = input_ids
    full_logprobs: list[float]
    loss_mask = [False] * token_count
    route: dict[str, Any]
    prefill_comparison: dict[str, Any] | None = None
    policy_identity_between: dict[str, Any] | None = None
    if prefill_rescore:
        decode_sampling_params = {
            "max_tokens": _PREFILL_RESCORE_COMPLETION_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": 1,
            "min_tokens": _PREFILL_RESCORE_COMPLETION_TOKENS,
        }
        decode_request: dict[str, Any] = {
            "model": model,
            "token_ids": list(input_ids),
            "sampling_params": decode_sampling_params,
            "cache_salt": cache_salt,
        }
        request_bodies["decode"] = decode_request
        decode_response = post_json(generate_endpoint, decode_request, timeout)
        response_hashes["decode"] = _json_sha256(decode_response)
        decode_choices = decode_response.get("choices")
        if (
            not isinstance(decode_choices, list)
            or len(decode_choices) != 1
            or not isinstance(decode_choices[0], dict)
        ):
            raise ReferenceError("decode response must contain one choice")
        decode_choice = decode_choices[0]
        raw_generated_ids = decode_choice.get("token_ids")
        if (
            not isinstance(raw_generated_ids, list)
            or len(raw_generated_ids) != _PREFILL_RESCORE_COMPLETION_TOKENS
            or any(isinstance(token, bool) or not isinstance(token, int) for token in raw_generated_ids)
        ):
            raise ReferenceError("decode response must contain exactly 256 token IDs")
        generated_ids = list(raw_generated_ids)
        finish_reason = decode_choice.get("finish_reason")
        generated_logprobs = _flatten_completion_logprobs(
            decode_choice.get("logprobs"), generated_ids
        )
        full_ids = input_ids + generated_ids
        decode_route = decode_choice.get("routed_experts")
        if not isinstance(decode_route, dict):
            raise ReferenceError("decode response is missing routed experts")
        decode_route = _normalise_route(
            decode_route, _PREFILL_RESCORE_SEQUENCE_TOKENS - 1
        )

        between_payload = get_json(readiness_url, timeout)
        policy_identity_between = _validate_policy_receipt(
            _extract_policy_receipt(between_payload, model),
            expected_identity,
            model,
        )
        if policy_identity_between != policy_identity_before:
            raise ReferenceError("active policy receipt changed between decode and prefill")

        rescore_sampling_params = {
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "prompt_logprobs": 1,
        }
        rescore_request: dict[str, Any] = {
            "model": model,
            "token_ids": list(full_ids),
            "sampling_params": rescore_sampling_params,
            "cache_salt": rescore_cache_salt,
        }
        request_bodies["prefill"] = rescore_request
        rescore_response = post_json(generate_endpoint, rescore_request, timeout)
        response_hashes["prefill"] = _json_sha256(rescore_response)
        rescore_choices = rescore_response.get("choices")
        if (
            not isinstance(rescore_choices, list)
            or len(rescore_choices) != 1
            or not isinstance(rescore_choices[0], dict)
        ):
            raise ReferenceError("prefill response must contain one choice")
        rescore_choice = rescore_choices[0]
        rescored_logprobs = _flatten_prompt_logprobs(
            rescore_response.get("prompt_logprobs"), full_ids
        )
        rescore_route = rescore_choice.get("routed_experts")
        if not isinstance(rescore_route, dict):
            raise ReferenceError("prefill response is missing routed experts")
        rescore_route = _normalise_route(
            rescore_route, _PREFILL_RESCORE_SEQUENCE_TOKENS
        )
        full_logprobs = [0.0] * _PREFILL_RESCORE_PROMPT_TOKENS + rescored_logprobs[
            _PREFILL_RESCORE_PROMPT_TOKENS:
        ]
        loss_mask = [False] * _PREFILL_RESCORE_PROMPT_TOKENS + [True] * _PREFILL_RESCORE_COMPLETION_TOKENS
        deltas = [
            prefill - decoded
            for decoded, prefill in zip(
                generated_logprobs,
                rescored_logprobs[_PREFILL_RESCORE_PROMPT_TOKENS:],
            )
        ]
        if any(not math.isfinite(value) for value in deltas):
            raise ReferenceError("decode-vs-prefill logprob delta is not finite")
        route_differences = _route_prediction_comparison(
            decode_route,
            rescore_route,
            prompt_tokens=_PREFILL_RESCORE_PROMPT_TOKENS,
            completion_tokens=_PREFILL_RESCORE_COMPLETION_TOKENS,
        )
        request_bodies = {"decode": decode_request, "prefill": rescore_request}
        prefill_comparison = {
            "decode_request": decode_request,
            "prefill_request": rescore_request,
            "rescore_request": rescore_request,
            "request_parameters": {
                "decode": decode_sampling_params,
                "prefill": rescore_sampling_params,
            },
            "salts": {
                "decode": cache_salt,
                "prefill": rescore_cache_salt,
                "rescore": rescore_cache_salt,
            },
            "response_sha256": {
                "decode": response_hashes["decode"],
                "prefill": response_hashes["prefill"],
                "rescore": response_hashes["prefill"],
            },
            "responses": {
                "decode": decode_response,
                "prefill": rescore_response,
                "rescore": rescore_response,
            },
            "route_sha256": {
                "decode": decode_route["sha256"],
                "prefill": rescore_route["sha256"],
                "rescore": rescore_route["sha256"],
            },
            "decode_route": decode_route,
            "prefill_route": rescore_route,
            "rescore_route": rescore_route,
            "completion_logprob_deltas": deltas,
            "per_completion": [
                {
                    "index": index,
                    "token_id": token_id,
                    "decode_logprob": decoded,
                    "prefill_logprob": rescored,
                    "delta": delta,
                }
                for index, (token_id, decoded, rescored, delta) in enumerate(
                    zip(
                        generated_ids,
                        generated_logprobs,
                        rescored_logprobs[_PREFILL_RESCORE_PROMPT_TOKENS:],
                        deltas,
                    )
                )
            ],
            "predicting_route_differences": route_differences,
        }
        route = rescore_route
        if expected_shape is not None and route.get("shape") != expected_shape:
            raise ReferenceError("prefill routed expert shape differs from the preserved expectation")
        if expected_digest is not None and route.get("sha256") != expected_digest:
            raise ReferenceError("prefill routed expert digest differs from the preserved expectation")
    else:
        sampling_params = {
            "max_tokens": completion_tokens if policy == "policy-0" else 1,
            "temperature": 0.0 if policy == "policy-0" else 1.0,
            "top_p": 1.0,
        }
        if policy == "base":
            sampling_params["prompt_logprobs"] = 1
        if policy == "policy-0":
            sampling_params["logprobs"] = 1
            sampling_params["min_tokens"] = completion_tokens
        request_body: dict[str, Any] = {
            "model": model,
            "token_ids": input_ids,
            "sampling_params": sampling_params,
        }
        if cache_salt is not None:
            request_body["cache_salt"] = cache_salt
        response = post_json(
            generate_endpoint if policy == "policy-0" else _endpoint_url(endpoint),
            request_body,
            timeout,
        )
        response_hashes["decode"] = _json_sha256(response)
        request_bodies["decode"] = request_body
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ReferenceError("serialized generate response must contain one choice")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        prompt_logprobs = (
            [0.0] * token_count
            if policy == "policy-0"
            else _flatten_prompt_logprobs(response.get("prompt_logprobs"), input_ids)
        )
        full_logprobs = prompt_logprobs
        if policy == "policy-0":
            raw_generated_ids = choice.get("token_ids")
            if (
                not isinstance(raw_generated_ids, list)
                or len(raw_generated_ids) < minimum_completion_tokens
                or len(raw_generated_ids) > completion_tokens
                or any(isinstance(token, bool) or not isinstance(token, int) for token in raw_generated_ids)
            ):
                raise ReferenceError("policy-0 response has too few or invalid generated token IDs")
            if len(input_ids) + len(raw_generated_ids) != trainer_seq_len:
                raise ReferenceError(
                    f"policy-0 response must contain exactly trainer seq_len={trainer_seq_len} tokens"
                )
            generated_ids = list(raw_generated_ids)
            generated_logprobs = _flatten_completion_logprobs(
                choice.get("logprobs"), generated_ids
            )
            full_ids = input_ids + generated_ids
            full_logprobs = prompt_logprobs + generated_logprobs
            loss_mask = [False] * token_count + [True] * len(generated_ids)
            if any(
                logprob != 0.0
                for logprob, trainable in zip(full_logprobs, loss_mask)
                if not trainable
            ):
                raise ReferenceError("policy-0 prompt inference logprobs must be zero")
        route = choice.get("routed_experts")
        if not isinstance(route, dict):
            raise ReferenceError("serialized generate response is missing routed experts")
        # vLLM emits one fewer route row than tokens for the default MTP0 path:
        # the final generated token has no following-token forward position.  The
        # trainer loader applies the same zero-tail padding as production rollout
        # packing; keep the serialized qualification artifact raw.
        route_token_count = len(full_ids) - 1 if policy == "policy-0" else len(full_ids)
        route = _normalise_route(route, route_token_count)
        if expected_shape is not None and route.get("shape") != expected_shape:
            raise ReferenceError("routed expert shape differs from the preserved expectation")
        if expected_digest is not None and route.get("sha256") != expected_digest:
            raise ReferenceError("routed expert digest differs from the preserved expectation")

    primary_request = "prefill" if prefill_rescore else "decode"
    sampling_params = request_bodies[primary_request]["sampling_params"]
    primary_cache_salt = (
        rescore_cache_salt if prefill_rescore else cache_salt
    )
    result = {
        "schema_version": 1,
        "source_kind": record["source_kind"],
        "input_ids": full_ids,
        "inference_logprobs": full_logprobs,
        "sequence_lengths": (
            [len(full_ids)] if policy == "policy-0" else list(record["sequence_lengths"])
        ),
        "routed_experts": route,
        "temperature": sampling_params["temperature"],
        "policy": policy,
        "model": model,
        "policy_identity": policy_identity,
        "cache_salt": primary_cache_salt,
        "request_mode": (
            _PREFILL_RESCORE_REQUEST_MODE
            if prefill_rescore
            else (
                "serialized_generate_completion_logprobs"
                if policy == "policy-0"
                else "serialized_generate_prompt_logprobs"
            )
        ),
        "sampling_params": sampling_params,
        "source_input_sha256": hashlib.sha256(
            json.dumps(input_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if policy == "policy-0":
        after_payload = get_json(readiness_url, timeout)
        policy_identity_after = _validate_policy_receipt(
            _extract_policy_receipt(after_payload, model),
            expected_identity,
            model,
        )
        if policy_identity_after != policy_identity_before:
            raise ReferenceError("active policy receipt changed during generation")
        result["policy_identity_before"] = policy_identity_before
        result["policy_identity_after"] = policy_identity_after
        if prefill_rescore:
            assert policy_identity_between is not None
            result["policy_identity_between"] = policy_identity_between
            result["rescore_cache_salt"] = rescore_cache_salt
            result["decode_cache_salt"] = cache_salt
            result["decode_route_sha256"] = prefill_comparison["route_sha256"]["decode"]
            result["rescore_route_sha256"] = prefill_comparison["route_sha256"]["prefill"]
            result["response_sha256"] = prefill_comparison["response_sha256"]
            result["prefill_rescore_comparison"] = prefill_comparison
            result["request_bodies"] = request_bodies
        result["loss_mask"] = loss_mask
        result["generated_token_ids"] = generated_ids
        result["generated_token_count"] = len(generated_ids)
        result["finish_reason"] = finish_reason
    serialized = json.dumps(result, separators=(",", ":")) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if prefill_rescore:
        temporary_path = output_path.with_name(
            f"{output_path.name}.partial.prefill-rescore.{os.getpid()}"
        )
        try:
            with temporary_path.open("x") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, output_path, follow_symlinks=False)
            except FileExistsError as error:
                raise ReferenceError(
                    f"refusing to clobber existing qualification output: {output_path}"
                ) from error
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(output_path.parent, directory_flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    else:
        output_path.write_text(serialized)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="full serialized /inference/v1/generate URL")
    parser.add_argument("--input-record", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--make-input-record", type=Path)
    parser.add_argument("--tokenizer", help="local tokenizer path for --make-input-record")
    parser.add_argument("--token-count", type=int, default=_DEFAULT_TOKEN_COUNT)
    parser.add_argument("--base-model")
    parser.add_argument("--policy", choices=sorted(_POLICIES))
    parser.add_argument("--policy-model", help="exact loaded LoRA model name; required for policy-0")
    parser.add_argument("--policy-receipt", type=Path, help="JSON readiness/load receipt for policy-0")
    parser.add_argument("--expected-policy-identity", type=Path, help="JSON expected policy identity for policy-0")
    parser.add_argument("--cache-salt", help="fresh top-level prefix-cache salt for policy-0")
    parser.add_argument(
        "--prefill-rescore",
        action="store_true",
        help="opt in to the isolated decode256 then full-sequence prefill-rescore qualification",
    )
    parser.add_argument(
        "--rescore-cache-salt",
        help="distinct fresh top-level prefix-cache salt for prefill-rescore",
    )
    parser.add_argument("--readiness-endpoint", help="explicit active staging readiness URL for policy-0")
    parser.add_argument("--completion-tokens", type=int)
    parser.add_argument("--minimum-completion-tokens", type=int)
    parser.add_argument("--trainer-seq-len", type=int)
    parser.add_argument("--minimum-tokens", type=int, default=4097)
    parser.add_argument("--expected-route-shape", type=int, nargs=3)
    parser.add_argument("--expected-route-sha256")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.make_input_record is not None:
            if not args.tokenizer:
                raise ReferenceError("--tokenizer is required with --make-input-record")
            write_input_record(
                args.make_input_record,
                args.tokenizer,
                token_count=args.token_count,
                prefill_rescore=args.prefill_rescore,
            )
            return 0
        if not all((args.endpoint, args.input_record, args.output, args.base_model, args.policy)):
            raise ReferenceError(
                "--endpoint, --input-record, --output, --base-model, and --policy are required"
            )
        build_reference(
            endpoint=args.endpoint,
            input_record=args.input_record,
            output_path=args.output,
            base_model=args.base_model,
            policy=args.policy,
            policy_model=args.policy_model,
            policy_receipt=args.policy_receipt,
            expected_policy_identity=args.expected_policy_identity,
            cache_salt=args.cache_salt,
            prefill_rescore=args.prefill_rescore,
            rescore_cache_salt=args.rescore_cache_salt,
            readiness_endpoint=args.readiness_endpoint,
            completion_tokens=args.completion_tokens,
            minimum_completion_tokens=args.minimum_completion_tokens,
            trainer_seq_len=args.trainer_seq_len,
            minimum_tokens=args.minimum_tokens,
            expected_route_shape=args.expected_route_shape,
            expected_route_sha256=args.expected_route_sha256,
            timeout=args.timeout,
        )
    except ReferenceError as error:
        print(f"reference refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
