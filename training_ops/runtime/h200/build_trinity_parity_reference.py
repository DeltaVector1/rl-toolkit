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
_DEFAULT_TOKEN_COUNT = 8192
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
    "config_sha256",
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
    encode: Callable[[str], list[int]] | None = None,
) -> dict[str, Any]:
    """Tokenize a fresh deterministic seed locally and write one sequence.

    ``local_files_only`` is used for the production tokenizer load. This mode
    never reads rollout files and never sends model data over SSH.
    """
    if isinstance(token_count, bool) or token_count < 4097:
        raise ReferenceError("input token_count must be at least 4097")
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


def _endpoint_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/inference/v1/generate"):
        raise ReferenceError("endpoint must be the serialized /inference/v1/generate URL")
    return endpoint


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
        if field in expected_identity and expected_identity[field] != value:
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
    for field in _POLICY_RECEIPT_MATCH_FIELDS:
        if expected_identity.get(field) != actual.get(field):
            raise ReferenceError(f"policy receipt differs in {field}")
    return actual


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
    minimum_tokens: int = 4097,
    expected_route_shape: list[int] | None = None,
    expected_route_sha256: str | None = None,
    timeout: float = 120.0,
    post_json: Callable[[str, Mapping[str, Any], float], dict[str, Any]] = _post_json,
) -> dict[str, Any]:
    if policy not in _POLICIES:
        raise ReferenceError(f"policy must be one of {sorted(_POLICIES)}")
    if not base_model or (policy == "policy-0" and not policy_model):
        raise ReferenceError("base_model and selected policy model are required")
    if isinstance(minimum_tokens, bool) or minimum_tokens < 0:
        raise ReferenceError("minimum_tokens must be non-negative")
    record = _load_record(input_record)
    input_ids = list(record["input_ids"])
    token_count = len(input_ids)
    if token_count < minimum_tokens:
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
        if policy_receipt is None or expected_policy_identity is None:
            raise ReferenceError(
                "policy-0 requires a readiness receipt and expected policy identity"
            )
        policy_identity = _validate_policy_receipt(
            policy_receipt,
            expected_policy_identity,
            model,
        )
    else:
        if policy_receipt is not None or expected_policy_identity is not None:
            raise ReferenceError("base policy cannot use a policy readiness receipt")
        policy_identity = None
    cache_salt = _validate_cache_salt(cache_salt, required=policy == "policy-0")
    sampling_params = {
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "prompt_logprobs": 1,
    }
    request_body: dict[str, Any] = {
        "model": model,
        "token_ids": input_ids,
        "sampling_params": sampling_params,
    }
    if cache_salt is not None:
        request_body["cache_salt"] = cache_salt
    response = post_json(
        _endpoint_url(endpoint),
        request_body,
        timeout,
    )
    logprobs = _flatten_prompt_logprobs(response.get("prompt_logprobs"), input_ids)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ReferenceError("serialized generate response must contain one choice")
    route = choices[0].get("routed_experts")
    if not isinstance(route, dict):
        raise ReferenceError("serialized generate response is missing routed experts")
    route = _normalise_route(route, token_count)
    if expected_shape is not None and route.get("shape") != expected_shape:
        raise ReferenceError("routed expert shape differs from the preserved expectation")
    if expected_digest is not None and route.get("sha256") != expected_digest:
        raise ReferenceError("routed expert digest differs from the preserved expectation")

    result = {
        "schema_version": 1,
        "source_kind": record["source_kind"],
        "input_ids": input_ids,
        "inference_logprobs": logprobs,
        "sequence_lengths": list(record["sequence_lengths"]),
        "routed_experts": route,
        "temperature": 1.0,
        "policy": policy,
        "model": model,
        "policy_identity": policy_identity,
        "cache_salt": cache_salt,
        "request_mode": "serialized_generate_prompt_logprobs",
        "sampling_params": sampling_params,
        "source_input_sha256": hashlib.sha256(
            json.dumps(input_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
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
