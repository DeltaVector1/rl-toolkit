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
import sys
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_RAW_ROUTE_FIELDS = {"data", "dtype", "shape", "start"}
_MAX_ROUTE_BYTES = 512 * 1024 * 1024
_POLICIES = {"base", "policy-0"}
_DEFAULT_TOKEN_COUNT = 8192
_NON_ROLLOUT_SOURCES = {
    "deterministic_local_tokenizer_seed",
    "hub_cache_non_rollout",
    "operator_non_rollout",
}


class ReferenceError(ValueError):
    """The preserved prompt or serialized generate response is unsafe to use."""


def _seed_text(blocks: int) -> str:
    """Return a deterministic, varied natural-language parity prompt."""
    return "\n".join(
        f"Parity block {index}: compare the exact token ledger, preserve every "
        f"route decision, and report whether the numbered condition {index % 97} "
        "remains stable under the selected policy."
        for index in range(blocks)
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
    if len(set(input_ids)) < 64:
        raise ReferenceError("deterministic seed did not produce diverse token IDs")
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


def build_reference(
    *,
    endpoint: str,
    input_record: Path | Mapping[str, Any],
    output_path: Path,
    base_model: str,
    policy: str,
    policy_model: str | None = None,
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
    response = post_json(
        _endpoint_url(endpoint),
        {
            "model": model,
            "token_ids": input_ids,
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "prompt_logprobs": 0,
                "extra_args": {
                    "prime_exact_target_prompt_logprobs": True,
                },
            },
        },
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
