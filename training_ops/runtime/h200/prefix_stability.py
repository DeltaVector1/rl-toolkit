"""Run a bounded H200 prefix-shape stability diagnostic.

The diagnostic uses only the fresh 512-token policy-0 reference.  It sends
five isolated serialized-generate requests, keeps the loaded policy receipt
fixed, and records route and logprob differences without applying a pass gate.
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
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from training_ops.runtime.h200.build_trinity_parity_reference import (
    _extract_policy_receipt,
    _get_json,
    _json_sha256,
    _normalise_route,
    _post_json,
    _validate_policy_receipt,
)
from training_ops.runtime.h200.prepare_trinity_policy0_qualification import (
    QualificationError,
    _validate_route_payload,
    validate_reference,
)


PREFIX_LENGTHS = (256, 257, 288, 384, 512)
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIAGNOSTIC_NAME = "h200_prefix_stability"


class PrefixStabilityError(ValueError):
    """The prefix-stability request or artifact is not safe to use."""


def _load_json(value: Path | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, Path):
        try:
            value = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PrefixStabilityError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        raise PrefixStabilityError(f"{label} must be a JSON object")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PrefixStabilityError(f"cannot read reference {path}") from error
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefixStabilityError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number == -9999.0:
        raise PrefixStabilityError(f"{label} is not finite evidence")
    return number


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise PrefixStabilityError(f"{label} is not a sha256 digest")
    return value


def _prompt_logprobs(response: Mapping[str, Any], token_ids: list[int]) -> list[float]:
    raw = response.get("prompt_logprobs")
    if not isinstance(raw, list) or len(raw) != len(token_ids):
        raise PrefixStabilityError("prompt logprob count does not match the exact prefix")
    result: list[float] = []
    for index, token_id in enumerate(token_ids):
        entry = raw[index]
        if index == 0:
            if entry is not None:
                raise PrefixStabilityError("leading prompt logprob must be null")
            result.append(0.0)
            continue
        if not isinstance(entry, Mapping):
            raise PrefixStabilityError(f"prompt logprob {index} is missing")
        candidates = [
            candidate
            for key, candidate in entry.items()
            if key == token_id or str(key) == str(token_id)
        ]
        if len(candidates) != 1 or not isinstance(candidates[0], Mapping):
            raise PrefixStabilityError(
                f"prompt logprob token key {index} does not match {token_id}"
            )
        result.append(_finite_number(candidates[0].get("logprob"), "prompt logprob"))
    return result


def _completion_logprob(response: Mapping[str, Any], token_id: int) -> float:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise PrefixStabilityError("decode response must contain one choice")
    content = choices[0].get("logprobs")
    content = content.get("content") if isinstance(content, Mapping) else None
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], Mapping):
        raise PrefixStabilityError("decode response must contain one completion logprob")
    entry = content[0]
    if entry.get("token") != f"token_id:{token_id}":
        raise PrefixStabilityError("decode completion token key is misaligned")
    return _finite_number(entry.get("logprob"), "decode completion logprob")


def _completion_logprobs(
    response: Mapping[str, Any], token_ids: list[int], label: str
) -> list[float]:
    choice = _choice(response)
    raw = choice.get("logprobs")
    content = raw.get("content") if isinstance(raw, Mapping) else None
    if not isinstance(content, list) or len(content) != len(token_ids):
        raise PrefixStabilityError(f"{label} completion logprob count is misaligned")
    result = []
    for index, token_id in enumerate(token_ids):
        entry = content[index]
        if not isinstance(entry, Mapping) or entry.get("token") != f"token_id:{token_id}":
            raise PrefixStabilityError(f"{label} completion token key is misaligned")
        result.append(_finite_number(entry.get("logprob"), f"{label} completion logprob"))
    return result


def _choice(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise PrefixStabilityError("serialized generate response must contain one choice")
    return choices[0]


def _route_rows(route: Mapping[str, Any]) -> list[list[int]]:
    shape = route.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise PrefixStabilityError("normalised route shape is invalid")
    try:
        raw = base64.b64decode(route["data"], validate=True)
    except (KeyError, binascii.Error, ValueError) as error:
        raise PrefixStabilityError("normalised route data is invalid") from error
    width = shape[1] * shape[2]
    if len(raw) != shape[0] * width:
        raise PrefixStabilityError("normalised route rows are misaligned")
    return [list(raw[offset : offset + width]) for offset in range(0, len(raw), width)]


def _route_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_shape = left.get("shape")
    right_shape = right.get("shape")
    left_dense_layers = left.get("dense_layers")
    right_dense_layers = right.get("dense_layers")
    if (
        not isinstance(left_shape, list)
        or not isinstance(right_shape, list)
        or len(left_shape) != 3
        or len(right_shape) != 3
        or left_shape[1:] != right_shape[1:]
        or left_dense_layers != right_dense_layers
    ):
        raise PrefixStabilityError("route comparison layer schemas differ")
    left_rows = _route_rows(left)
    right_rows = _route_rows(right)
    compared = min(len(left_rows), len(right_rows))
    differences: list[dict[str, Any]] = []
    different_set_layer_count = 0
    order_only_layer_count = 0
    different_set_row_count = 0
    order_only_row_count = 0
    sparse_layers = range(left_dense_layers, left_shape[1])
    top_k = left_shape[2]
    for row in range(compared):
        left_values = left_rows[row]
        right_values = right_rows[row]
        layer_differences = []
        row_has_set_difference = False
        row_has_order_only_difference = False
        for layer in sparse_layers:
            start = layer * top_k
            left_layer = left_values[start : start + top_k]
            right_layer = right_values[start : start + top_k]
            route_set_equal = set(left_layer) == set(right_layer)
            route_order_equal = left_layer == right_layer
            if not route_set_equal or not route_order_equal:
                if route_set_equal:
                    order_only_layer_count += 1
                    row_has_order_only_difference = True
                else:
                    different_set_layer_count += 1
                    row_has_set_difference = True
                layer_differences.append(
                    {
                        "layer": layer,
                        "left": left_layer,
                        "right": right_layer,
                        "route_set_equal": route_set_equal,
                        "route_order_equal": route_order_equal,
                    }
                )
        if row_has_set_difference:
            different_set_row_count += 1
        elif row_has_order_only_difference:
            order_only_row_count += 1
        if layer_differences:
            differences.append(
                {
                    "row": row,
                    "left": left_values,
                    "right": right_values,
                    "route_set_equal": not row_has_set_difference,
                    "route_order_equal": not (
                        row_has_set_difference or row_has_order_only_difference
                    ),
                    "layers": layer_differences,
                }
            )
    return {
        "left": left_label,
        "right": right_label,
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "compared_rows": compared,
        "different_row_count": len(differences),
        "different_set_row_count": different_set_row_count,
        "order_only_row_count": order_only_row_count,
        "different_set_layer_count": different_set_layer_count,
        "order_only_layer_count": order_only_layer_count,
        "different_layer_rows": differences,
        "different_rows": differences,
    }


def _score_comparison(
    left: list[float],
    right: list[float],
    left_label: str,
    right_label: str,
    *,
    start: int = 0,
) -> dict[str, Any]:
    compared = min(len(left), len(right)) - start
    if compared < 0:
        compared = 0
    differences = []
    for position in range(start, start + compared):
        delta = left[position] - right[position]
        if delta != 0.0:
            differences.append(
                {
                    "position": position,
                    "left": left[position],
                    "right": right[position],
                    "delta": delta,
                }
            )
    return {
        "left": left_label,
        "right": right_label,
        "compared_positions": compared,
        "different_position_count": len(differences),
        "different_positions": differences,
    }


def _comparison_collections(
    requests: list[Mapping[str, Any]],
    decode_route: Mapping[str, Any],
    rescore_route: Mapping[str, Any],
    rescore_logprobs: list[float],
) -> dict[str, list[dict[str, Any]]]:
    route_pairs = []
    for left_index in range(len(requests)):
        for right_index in range(left_index + 1, len(requests)):
            route_pairs.append(
                _route_comparison(
                    requests[left_index]["route"],
                    requests[right_index]["route"],
                    f"prefix_{requests[left_index]['prefix_length']}",
                    f"prefix_{requests[right_index]['prefix_length']}",
                )
            )
    original_decode_routes = []
    original_rescore_routes = []
    for request in requests:
        original_decode_routes.append(
            _route_comparison(
                request["route"],
                decode_route,
                f"prefix_{request['prefix_length']}",
                "original_decode",
            )
        )
        original_rescore_routes.append(
            _route_comparison(
                request["route"],
                rescore_route,
                f"prefix_{request['prefix_length']}",
                "original_rescore",
            )
        )
    prompt_requests = [request for request in requests if request["prefix_length"] > 256]
    score_pairs = []
    for left_index in range(len(prompt_requests)):
        for right_index in range(left_index + 1, len(prompt_requests)):
            score_pairs.append(
                _score_comparison(
                    prompt_requests[left_index]["prompt_logprobs"],
                    prompt_requests[right_index]["prompt_logprobs"],
                    f"prefix_{prompt_requests[left_index]['prefix_length']}",
                    f"prefix_{prompt_requests[right_index]['prefix_length']}",
                    start=1,
                )
            )
    original_score_comparisons = []
    for request in prompt_requests:
        original_score_comparisons.append(
            _score_comparison(
                request["prompt_logprobs"],
                rescore_logprobs,
                f"prefix_{request['prefix_length']}",
                "original_rescore",
                start=1,
            )
        )
    return {
        "common_prefix_routes": route_pairs,
        "original_decode_routes": original_decode_routes,
        "original_rescore_routes": original_rescore_routes,
        "common_prefix_logprobs": score_pairs,
        "original_rescore_logprobs": original_score_comparisons,
    }


def _original_predicting_route_differences(
    decode_route: Mapping[str, Any], rescore_route: Mapping[str, Any]
) -> dict[str, Any]:
    decode_rows = _route_rows(decode_route)
    rescore_rows = _route_rows(rescore_route)
    differences = []
    for row in range(255, 511):
        if decode_rows[row] != rescore_rows[row]:
            differences.append(
                {
                    "row": row,
                    "decode": decode_rows[row],
                    "prefill": rescore_rows[row],
                }
            )
    return {
        "first_predicting_row": 255,
        "last_predicting_row_exclusive": 511,
        "compared_rows": 256,
        "different_rows": differences,
        "different_row_count": len(differences),
    }


def _validate_reference_for_diagnostic(
    reference: Mapping[str, Any],
    expected_policy_identity: Mapping[str, Any],
    policy_model: str,
) -> dict[str, Any]:
    try:
        validate_reference(reference)
    except QualificationError as error:
        raise PrefixStabilityError(f"reference contract is invalid: {error}") from error
    input_ids = reference.get("input_ids")
    if (
        reference.get("request_mode") != "serialized_generate_decode256_prefill_rescore"
        or reference.get("policy") != "policy-0"
        or reference.get("model") != policy_model
        or input_ids != list(input_ids or [])
        or len(input_ids) != 512
        or reference.get("sequence_lengths") != [512]
    ):
        raise PrefixStabilityError("reference is not the exact fresh 512-token policy reference")
    receipts = [
        reference.get("policy_identity"),
        reference.get("policy_identity_before"),
        reference.get("policy_identity_between"),
        reference.get("policy_identity_after"),
    ]
    if any(not isinstance(receipt, Mapping) for receipt in receipts):
        raise PrefixStabilityError("reference does not contain a complete policy receipt")
    try:
        expected = _validate_policy_receipt(
            receipts[0], expected_policy_identity, policy_model
        )
    except (ValueError, TypeError) as error:
        raise PrefixStabilityError(f"reference policy receipt is invalid: {error}") from error
    if any(dict(receipt) != expected for receipt in receipts[1:]):
        raise PrefixStabilityError("reference policy receipt is not identical at every boundary")
    comparison = reference.get("prefill_rescore_comparison")
    if not isinstance(comparison, Mapping):
        raise PrefixStabilityError("reference prefill-rescore evidence is missing")
    responses = comparison.get("responses")
    if not isinstance(responses, Mapping):
        raise PrefixStabilityError("reference exact responses are missing")
    decode_response = responses.get("decode")
    rescore_response = responses.get("rescore", responses.get("prefill"))
    if not isinstance(decode_response, Mapping) or not isinstance(rescore_response, Mapping):
        raise PrefixStabilityError("reference decode/rescore responses are missing")
    generated_ids = reference.get("generated_token_ids")
    if not isinstance(generated_ids, list) or len(generated_ids) != 256:
        raise PrefixStabilityError("reference generated suffix is not exactly 256 tokens")
    original_decode_logprobs = []
    decode_choices = _choice(decode_response)
    decode_token_ids = decode_choices.get("token_ids")
    if decode_token_ids != generated_ids:
        raise PrefixStabilityError("reference decode tokens do not match the generated suffix")
    raw_content = decode_choices.get("logprobs")
    raw_content = raw_content.get("content") if isinstance(raw_content, Mapping) else None
    if not isinstance(raw_content, list) or len(raw_content) != 256:
        raise PrefixStabilityError("reference decode logprobs are not exactly aligned")
    for index, token_id in enumerate(generated_ids):
        entry = raw_content[index]
        if not isinstance(entry, Mapping) or entry.get("token") != f"token_id:{token_id}":
            raise PrefixStabilityError("reference decode token key is misaligned")
        original_decode_logprobs.append(
            _finite_number(entry.get("logprob"), "reference decode logprob")
        )
    original_rescore_logprobs = _prompt_logprobs(rescore_response, input_ids)
    decode_route = comparison.get("decode_route")
    rescore_route = reference.get("routed_experts")
    if not isinstance(decode_route, Mapping) or not isinstance(rescore_route, Mapping):
        raise PrefixStabilityError("reference decode/rescore routes are missing")
    if decode_route.get("shape", [None])[0] != 511 or rescore_route.get("shape", [None])[0] != 512:
        raise PrefixStabilityError("reference decode/rescore route rows are not exact")
    return {
        "input_ids": list(input_ids),
        "receipt": expected,
        "comparison": comparison,
        "decode_response": dict(decode_response),
        "rescore_response": dict(rescore_response),
        "decode_logprobs": original_decode_logprobs,
        "rescore_logprobs": original_rescore_logprobs,
        "decode_route": dict(decode_route),
        "rescore_route": dict(rescore_route),
        "generated_ids": list(generated_ids),
    }


def _validate_endpoint(endpoint: str, label: str) -> str:
    if not isinstance(endpoint, str) or not endpoint:
        raise PrefixStabilityError(f"{label} is required")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PrefixStabilityError(f"{label} must be an HTTP URL")
    return endpoint.rstrip("/")


def _validate_endpoints(endpoint: str, readiness_endpoint: str) -> tuple[str, str]:
    generated = _validate_endpoint(endpoint, "serialized generate endpoint")
    readiness = _validate_endpoint(readiness_endpoint, "readiness endpoint")
    generated_parts = urlsplit(generated)
    readiness_parts = urlsplit(readiness)
    if not generated_parts.path.endswith("/inference/v1/generate"):
        raise PrefixStabilityError(
            "serialized generate endpoint must be the /inference/v1/generate URL"
        )
    if (
        readiness_parts.scheme != generated_parts.scheme
        or readiness_parts.netloc != generated_parts.netloc
        or not readiness_parts.path
        or readiness_parts.path == generated_parts.path
    ):
        raise PrefixStabilityError(
            "readiness endpoint must match the generate endpoint origin and use a distinct path"
        )
    return generated, readiness


def _salt(prefix: str, length: int) -> str:
    if not isinstance(prefix, str) or not prefix.strip() or len(prefix) > 220:
        raise PrefixStabilityError("fresh salt prefix must be non-empty and at most 220 characters")
    value = f"{prefix}-{length}"
    if len(value) > 256:
        raise PrefixStabilityError("fresh salt is too long")
    return value


def _request_for(
    input_ids: list[int],
    policy_model: str,
    prefix_length: int,
    fresh_salt_prefix: str,
) -> dict[str, Any]:
    if prefix_length == 256:
        sampling_params = {
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": 1,
            "min_tokens": 1,
        }
    else:
        sampling_params = {
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "prompt_logprobs": 1,
        }
    return {
        "model": policy_model,
        "token_ids": list(input_ids[:prefix_length]),
        "sampling_params": sampling_params,
        "cache_salt": _salt(fresh_salt_prefix, prefix_length),
    }


def _validate_response(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    prefix_length: int,
    expected_first_token: int,
) -> tuple[dict[str, Any], list[float], int | None, float]:
    token_ids = request.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != prefix_length:
        raise PrefixStabilityError("request token prefix is misaligned")
    choice = _choice(response)
    raw_route = choice.get("routed_experts")
    if not isinstance(raw_route, Mapping):
        raise PrefixStabilityError("serialized response is missing routed experts")
    expected_route_rows = prefix_length
    try:
        route = _normalise_route(raw_route, expected_route_rows)
    except (ValueError, TypeError) as error:
        raise PrefixStabilityError(f"response route is invalid: {error}") from error
    if prefix_length == 256:
        generated = choice.get("token_ids")
        if (
            not isinstance(generated, list)
            or len(generated) != 1
            or isinstance(generated[0], bool)
            or not isinstance(generated[0], int)
            or generated[0] != expected_first_token
        ):
            raise PrefixStabilityError("256-token decode did not reproduce the original first token")
        score = _completion_logprob(response, expected_first_token)
        return route, [], generated[0], score
    prompt_scores = _prompt_logprobs(response, token_ids)
    return route, prompt_scores, None, prompt_scores[-1]


def _receipt_event(
    get_json: Callable[[str, float], dict[str, Any]],
    readiness_endpoint: str,
    expected: Mapping[str, Any],
    policy_model: str,
    timeout: float,
    stage: str,
    request_index: int,
) -> dict[str, Any]:
    try:
        payload = get_json(readiness_endpoint, timeout)
        actual = _extract_policy_receipt(payload, policy_model)
        receipt = _validate_policy_receipt(actual, expected, policy_model)
    except (ValueError, TypeError) as error:
        raise PrefixStabilityError(f"readiness receipt at {stage} is invalid: {error}") from error
    if dict(receipt) != dict(expected):
        raise PrefixStabilityError(f"policy receipt changed at {stage}")
    return {"stage": stage, "request_index": request_index, "receipt": dict(receipt)}


def _artifact_original_evidence(
    value: Mapping[str, Any], input_ids: list[int]
) -> dict[str, Any]:
    original = value.get("original_evidence")
    if not isinstance(original, Mapping):
        raise PrefixStabilityError("diagnostic original evidence is missing")
    decode_response = original.get("decode_response")
    rescore_response = original.get("rescore_response")
    if not isinstance(decode_response, Mapping) or not isinstance(rescore_response, Mapping):
        raise PrefixStabilityError("diagnostic original responses are missing")
    decode_choice = _choice(decode_response)
    generated_ids = decode_choice.get("token_ids")
    if (
        not isinstance(generated_ids, list)
        or len(generated_ids) != 256
        or any(isinstance(token, bool) or not isinstance(token, int) for token in generated_ids)
    ):
        raise PrefixStabilityError("original decode token IDs are invalid")
    decode_logprobs = _completion_logprobs(
        decode_response, generated_ids, "original decode"
    )
    rescore_logprobs = _prompt_logprobs(rescore_response, input_ids)
    try:
        decode_route = _normalise_route(
            decode_choice["routed_experts"], 511
        )
        rescore_route = _normalise_route(
            _choice(rescore_response)["routed_experts"], 512
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PrefixStabilityError("original response route is invalid") from error
    expected = {
        "decode_response_sha256": _json_sha256(decode_response),
        "rescore_response_sha256": _json_sha256(rescore_response),
        "decode_route_sha256": decode_route["sha256"],
        "rescore_route_sha256": rescore_route["sha256"],
        "first_decoded_token_id": generated_ids[0],
        "first_decoded_logprob": decode_logprobs[0],
        "first_rescored_actual_token_logprob": rescore_logprobs[256],
        "reference_route_differences": _original_predicting_route_differences(
            decode_route, rescore_route
        ),
    }
    for key, expected_value in expected.items():
        if original.get(key) != expected_value:
            raise PrefixStabilityError(f"original evidence field {key} is not recomputed")
    return {
        "decode_response": decode_response,
        "rescore_response": rescore_response,
        "generated_ids": generated_ids,
        "decode_logprobs": decode_logprobs,
        "rescore_logprobs": rescore_logprobs,
        "decode_route": decode_route,
        "rescore_route": rescore_route,
    }


def validate_diagnostic(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate artifact structure, hashes, exact token prefixes, and evidence."""
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise PrefixStabilityError("diagnostic schema version is invalid")
    if value.get("diagnostic") != _DIAGNOSTIC_NAME:
        raise PrefixStabilityError("diagnostic name is invalid")
    prefix_lengths = value.get("prefix_lengths")
    if prefix_lengths != list(PREFIX_LENGTHS):
        raise PrefixStabilityError("diagnostic prefix lengths are invalid")
    input_ids = value.get("reference_input_ids")
    if (
        not isinstance(input_ids, list)
        or len(input_ids) != 512
        or any(isinstance(token, bool) or not isinstance(token, int) for token in input_ids)
    ):
        raise PrefixStabilityError("diagnostic reference input IDs are invalid")
    _validate_digest(value.get("reference_sha256"), "reference hash")
    policy_model = value.get("policy_model")
    fresh_salt_prefix = value.get("fresh_salt_prefix")
    if not isinstance(policy_model, str) or not policy_model:
        raise PrefixStabilityError("diagnostic policy model is invalid")
    if not isinstance(fresh_salt_prefix, str) or not fresh_salt_prefix.strip():
        raise PrefixStabilityError("diagnostic salt prefix is invalid")
    expected_identity = value.get("expected_policy_identity")
    if not isinstance(expected_identity, Mapping):
        raise PrefixStabilityError("diagnostic policy identity is missing")
    try:
        _validate_policy_receipt(expected_identity, expected_identity, policy_model)
    except (ValueError, TypeError) as error:
        raise PrefixStabilityError("diagnostic policy identity is invalid") from error
    original = value.get("original_evidence")
    original_info = _artifact_original_evidence(value, input_ids)
    expected_first_token = original_info["generated_ids"][0]
    requests = value.get("requests")
    if not isinstance(requests, list) or len(requests) != len(PREFIX_LENGTHS):
        raise PrefixStabilityError("diagnostic request count is invalid")
    for index, entry in enumerate(requests):
        if not isinstance(entry, Mapping) or entry.get("prefix_length") != PREFIX_LENGTHS[index]:
            raise PrefixStabilityError("diagnostic request prefix ordering is invalid")
        request = entry.get("request")
        response = entry.get("response")
        if not isinstance(request, Mapping) or not isinstance(response, Mapping):
            raise PrefixStabilityError("diagnostic raw request or response is missing")
        if request.get("token_ids") != input_ids[: PREFIX_LENGTHS[index]]:
            raise PrefixStabilityError("diagnostic request token prefix is misaligned")
        expected_request = _request_for(
            input_ids, policy_model, PREFIX_LENGTHS[index], fresh_salt_prefix
        )
        if dict(request) != expected_request:
            raise PrefixStabilityError("diagnostic request parameters differ")
        if entry.get("salt") != request.get("cache_salt"):
            raise PrefixStabilityError("diagnostic request salt is not bound")
        if _json_sha256(request) != entry.get("request_sha256"):
            raise PrefixStabilityError("diagnostic request hash differs")
        if _json_sha256(response) != entry.get("response_sha256"):
            raise PrefixStabilityError("diagnostic response hash differs")
        route = entry.get("route")
        if not isinstance(route, Mapping):
            raise PrefixStabilityError("diagnostic route is missing")
        try:
            _validate_route_payload(route, PREFIX_LENGTHS[index], "diagnostic")
        except (ValueError, TypeError) as error:
            raise PrefixStabilityError("diagnostic route is invalid") from error
        if entry.get("route_sha256") != route.get("sha256"):
            raise PrefixStabilityError("diagnostic route hash or envelope differs")
        response_route, prompt_scores, generated_token_id, actual_score = _validate_response(
            response,
            request,
            PREFIX_LENGTHS[index],
            expected_first_token,
        )
        if (
            response_route != route
            or prompt_scores != entry.get("prompt_logprobs")
            or generated_token_id != entry.get("generated_token_id")
            or actual_score != entry.get("actual_token_logprob")
        ):
            raise PrefixStabilityError("diagnostic response evidence is not bound")
        if entry.get("actual_token_logprob") is not None:
            _finite_number(entry["actual_token_logprob"], "diagnostic actual-token logprob")
    first_request = requests[0]
    expected_original = {
        "first_decode_score_delta": (
            first_request["actual_token_logprob"] - original_info["decode_logprobs"][0]
        ),
        "first_rescore_score_delta": (
            first_request["actual_token_logprob"] - original_info["rescore_logprobs"][256]
        ),
    }
    for key, expected_value in expected_original.items():
        if original.get(key) != expected_value or not math.isfinite(float(expected_value)):
            raise PrefixStabilityError(f"original evidence field {key} is not recomputed")
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise PrefixStabilityError("diagnostic comparisons are missing")
    expected_comparisons = _comparison_collections(
        requests,
        original_info["decode_route"],
        original_info["rescore_route"],
        original_info["rescore_logprobs"],
    )
    if set(comparisons) != set(expected_comparisons):
        raise PrefixStabilityError("diagnostic comparison collections differ")
    for key, expected_value in expected_comparisons.items():
        if comparisons.get(key) != expected_value:
            raise PrefixStabilityError(f"diagnostic comparison collection {key} differs")
    sequence = value.get("receipt_sequence")
    if not isinstance(sequence, list) or len(sequence) != len(PREFIX_LENGTHS) + 1:
        raise PrefixStabilityError("diagnostic receipt sequence is incomplete")
    receipts = [entry.get("receipt") for entry in sequence if isinstance(entry, Mapping)]
    if (
        len(receipts) != len(sequence)
        or not receipts
        or any(receipt != receipts[0] for receipt in receipts[1:])
        or any(receipt != expected_identity for receipt in receipts)
    ):
        raise PrefixStabilityError("diagnostic receipt sequence changed")
    return dict(value)


def build_diagnostic(
    *,
    reference_path: Path,
    endpoint: str,
    readiness_endpoint: str,
    expected_policy_identity: Path | Mapping[str, Any],
    policy_model: str,
    fresh_salt_prefix: str,
    timeout: float = 120.0,
    post_json: Callable[[str, Mapping[str, Any], float], dict[str, Any]] = _post_json,
    get_json: Callable[[str, float], dict[str, Any]] = _get_json,
) -> dict[str, Any]:
    """Run the five bounded requests and return an unpublished artifact."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise PrefixStabilityError("timeout must be positive")
    endpoint, readiness_endpoint = _validate_endpoints(endpoint, readiness_endpoint)
    if not isinstance(policy_model, str) or not policy_model:
        raise PrefixStabilityError("exact policy model is required")
    reference = _load_json(reference_path, "reference")
    expected_identity = _load_json(expected_policy_identity, "expected policy identity")
    reference_info = _validate_reference_for_diagnostic(
        reference, expected_identity, policy_model
    )
    input_ids = reference_info["input_ids"]
    requests: list[dict[str, Any]] = []
    receipt_sequence: list[dict[str, Any]] = []
    for index, prefix_length in enumerate(PREFIX_LENGTHS):
        if index == 0:
            receipt_sequence.append(
                _receipt_event(
                    get_json,
                    readiness_endpoint,
                    reference_info["receipt"],
                    policy_model,
                    float(timeout),
                    "before",
                    index,
                )
            )
        request = _request_for(input_ids, policy_model, prefix_length, fresh_salt_prefix)
        try:
            response = post_json(endpoint, request, float(timeout))
        except Exception as error:
            raise PrefixStabilityError(f"prefix {prefix_length} request failed: {error}") from error
        if not isinstance(response, Mapping):
            raise PrefixStabilityError(f"prefix {prefix_length} response is not an object")
        route, prompt_scores, generated_token_id, actual_score = _validate_response(
            response,
            request,
            prefix_length,
            reference_info["generated_ids"][0],
        )
        requests.append(
            {
                "prefix_length": prefix_length,
                "salt": request["cache_salt"],
                "request": dict(request),
                "request_sha256": _json_sha256(request),
                "response": dict(response),
                "response_sha256": _json_sha256(response),
                "route": route,
                "route_sha256": route["sha256"],
                "prompt_logprobs": prompt_scores,
                "generated_token_id": generated_token_id,
                "actual_token_position": (
                    prefix_length - 1 if prefix_length > 256 else prefix_length
                ),
                "actual_token_logprob": actual_score,
            }
        )
        if index == len(PREFIX_LENGTHS) - 1:
            receipt_sequence.append(
                _receipt_event(
                    get_json,
                    readiness_endpoint,
                    reference_info["receipt"],
                    policy_model,
                    float(timeout),
                    "after",
                    index,
                )
            )
        else:
            receipt_sequence.append(
                _receipt_event(
                    get_json,
                    readiness_endpoint,
                    reference_info["receipt"],
                    policy_model,
                    float(timeout),
                    "between",
                    index,
                )
            )
    comparisons = _comparison_collections(
        requests,
        reference_info["decode_route"],
        reference_info["rescore_route"],
        reference_info["rescore_logprobs"],
    )
    first_request = requests[0]
    original_first_decode = reference_info["decode_logprobs"][0]
    original_first_rescore = reference_info["rescore_logprobs"][256]
    result = {
        "schema_version": 1,
        "diagnostic": _DIAGNOSTIC_NAME,
        "reference_sha256": _sha256_file(reference_path),
        "reference_input_ids": input_ids,
        "reference_input_sha256": reference.get("source_input_sha256"),
        "policy_model": policy_model,
        "expected_policy_identity": dict(reference_info["receipt"]),
        "fresh_salt_prefix": fresh_salt_prefix,
        "prefix_lengths": list(PREFIX_LENGTHS),
        "receipt_sequence": receipt_sequence,
        "requests": requests,
        "comparisons": comparisons,
        "original_evidence": {
            "decode_response": reference_info["decode_response"],
            "rescore_response": reference_info["rescore_response"],
            "decode_response_sha256": _json_sha256(reference_info["decode_response"]),
            "rescore_response_sha256": _json_sha256(reference_info["rescore_response"]),
            "decode_route_sha256": reference_info["decode_route"]["sha256"],
            "rescore_route_sha256": reference_info["rescore_route"]["sha256"],
            "first_decoded_token_id": reference_info["generated_ids"][0],
            "first_decoded_logprob": original_first_decode,
            "first_rescored_actual_token_logprob": original_first_rescore,
            "first_decode_score_delta": first_request["actual_token_logprob"] - original_first_decode,
            "first_rescore_score_delta": first_request["actual_token_logprob"] - original_first_rescore,
            "reference_route_differences": _original_predicting_route_differences(
                reference_info["decode_route"], reference_info["rescore_route"]
            ),
        },
    }
    validate_diagnostic(result)
    return result


def publish_diagnostic(value: Mapping[str, Any], output_path: Path) -> None:
    """Publish one artifact atomically and refuse every existing target."""
    validate_diagnostic(value)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output_path):
        raise PrefixStabilityError(f"refusing to clobber existing output: {output_path}")
    temporary_path = output_path.with_name(
        f"{output_path.name}.partial.prefix-stability.{os.getpid()}"
    )
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    created = False
    try:
        try:
            with temporary_path.open("x", encoding="utf-8") as stream:
                created = True
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as error:
            raise PrefixStabilityError(
                f"temporary output already exists; refusing to remove it: {temporary_path}"
            ) from error
        try:
            os.link(temporary_path, output_path, follow_symlinks=False)
        except FileExistsError as error:
            raise PrefixStabilityError(
                f"refusing to clobber existing output: {output_path}"
            ) from error
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(output_path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if created:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def run_diagnostic(
    *,
    reference_path: Path,
    endpoint: str,
    readiness_endpoint: str,
    expected_policy_identity: Path | Mapping[str, Any],
    policy_model: str,
    fresh_salt_prefix: str,
    output_path: Path,
    timeout: float = 120.0,
    post_json: Callable[[str, Mapping[str, Any], float], dict[str, Any]] = _post_json,
    get_json: Callable[[str, float], dict[str, Any]] = _get_json,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if os.path.lexists(output_path):
        raise PrefixStabilityError(f"refusing to clobber existing output: {output_path}")
    result = build_diagnostic(
        reference_path=Path(reference_path),
        endpoint=endpoint,
        readiness_endpoint=readiness_endpoint,
        expected_policy_identity=expected_policy_identity,
        policy_model=policy_model,
        fresh_salt_prefix=fresh_salt_prefix,
        timeout=timeout,
        post_json=post_json,
        get_json=get_json,
    )
    publish_diagnostic(result, output_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", "--reference-path", dest="reference", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        "--generate-endpoint",
        "--serialized-generate-endpoint",
        dest="endpoint",
        required=True,
    )
    parser.add_argument("--readiness-endpoint", required=True)
    parser.add_argument("--expected-policy-identity", type=Path, required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--fresh-salt-prefix", "--cache-salt-prefix", required=True)
    parser.add_argument("--output", "--output-path", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_diagnostic(
            reference_path=args.reference,
            endpoint=args.endpoint,
            readiness_endpoint=args.readiness_endpoint,
            expected_policy_identity=args.expected_policy_identity,
            policy_model=args.policy_model,
            fresh_salt_prefix=args.fresh_salt_prefix,
            output_path=args.output,
            timeout=args.timeout,
        )
    except (OSError, PrefixStabilityError) as error:
        print(f"prefix stability failed: {error}", file=sys.stderr)
        return 2
    print(f"prefix stability written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
