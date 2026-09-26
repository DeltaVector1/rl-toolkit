"""Validate the exact step-0 Trinity policy reference contract.

This module is local and side-effect free.  ``training_ops`` uses its
validators before it stages the three files through its existing fan-out path.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class QualificationError(ValueError):
    """The policy-0 qualification artifact is not safe to stage."""


_PREFILL_RESCORE_REQUEST_MODE = "serialized_generate_decode256_prefill_rescore"


def _validate_route_payload(
    value: Mapping[str, Any],
    expected_rows: int,
    label: str,
) -> None:
    if set(value) != {
        "schema_version",
        "encoding",
        "dtype",
        "shape",
        "dense_layers",
        "experts",
        "sha256",
        "data",
    }:
        raise QualificationError(f"{label} route schema is invalid")
    shape = value.get("shape")
    dense_layers = value.get("dense_layers")
    experts = value.get("experts")
    if (
        value.get("schema_version") != 1
        or value.get("encoding") != "base64"
        or value.get("dtype") != "uint8"
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        or shape[0] != expected_rows
        or isinstance(dense_layers, bool)
        or not isinstance(dense_layers, int)
        or not 0 <= dense_layers < shape[1]
        or isinstance(experts, bool)
        or not isinstance(experts, int)
        or not 1 <= experts <= 256
    ):
        raise QualificationError(f"{label} route metadata is invalid")
    encoded = value.get("data")
    digest = value.get("sha256")
    if (
        not isinstance(encoded, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise QualificationError(f"{label} route checksum metadata is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise QualificationError(f"{label} route data is invalid") from error
    if len(raw) != shape[0] * shape[1] * shape[2] or hashlib.sha256(raw).hexdigest() != digest:
        raise QualificationError(f"{label} route checksum is invalid")
    row_width = shape[1] * shape[2]
    for offset in range(0, len(raw), row_width):
        row = raw[offset : offset + row_width]
        dense_end = dense_layers * shape[2]
        if any(row[:dense_end]):
            raise QualificationError(f"{label} dense route IDs are invalid")
        for layer in range(dense_layers, shape[1]):
            selected = row[layer * shape[2] : (layer + 1) * shape[2]]
            if any(expert >= experts for expert in selected) or len(set(selected)) != len(selected):
                raise QualificationError(f"{label} sparse route IDs are invalid")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _strict_logprob(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise QualificationError(f"{label} is not finite")
    return value


def _completion_evidence(response: Mapping[str, Any], generated_ids: list[int]) -> list[float]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise QualificationError("prefill-rescore completion response is invalid")
    choice = choices[0]
    if choice.get("token_ids") != generated_ids:
        raise QualificationError("prefill-rescore completion token IDs differ")
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    if not isinstance(content, list) or len(content) != len(generated_ids):
        raise QualificationError("prefill-rescore completion logprob count differs")
    result = []
    for index, token_id in enumerate(generated_ids):
        entry = content[index]
        if not isinstance(entry, Mapping) or entry.get("token") != f"token_id:{token_id}":
            raise QualificationError("prefill-rescore completion token key differs")
        result.append(_strict_logprob(entry.get("logprob"), "completion logprob"))
    return result


def _prompt_evidence(response: Mapping[str, Any], token_ids: list[int]) -> list[float]:
    raw = response.get("prompt_logprobs")
    if not isinstance(raw, list) or len(raw) != len(token_ids):
        raise QualificationError("prefill-rescore prompt logprob count differs")
    result = []
    for index, token_id in enumerate(token_ids):
        entry = raw[index]
        if index == 0 and entry is None:
            result.append(0.0)
            continue
        if not isinstance(entry, Mapping):
            raise QualificationError(f"prefill-rescore prompt logprob {index} is missing")
        candidates = [
            candidate
            for key, candidate in entry.items()
            if key == token_id or str(key) == str(token_id)
        ]
        if len(candidates) != 1 or not isinstance(candidates[0], Mapping):
            raise QualificationError(
                f"prefill-rescore prompt token key {index} does not match {token_id}"
            )
        result.append(_strict_logprob(candidates[0].get("logprob"), "prompt logprob"))
    return result


def _route_prediction_differences(
    decode_route: Mapping[str, Any],
    prefill_route: Mapping[str, Any],
) -> dict[str, Any]:
    decode_shape = decode_route["shape"]
    prefill_shape = prefill_route["shape"]
    decode_raw = base64.b64decode(decode_route["data"], validate=True)
    prefill_raw = base64.b64decode(prefill_route["data"], validate=True)
    row_width = decode_shape[1] * decode_shape[2]
    if prefill_shape[1:] != decode_shape[1:] or row_width != prefill_shape[1] * prefill_shape[2]:
        raise QualificationError("prefill-rescore route row schemas differ")
    differences = []
    for row in range(255, 511):
        decode_ids = list(decode_raw[row * row_width : (row + 1) * row_width])
        prefill_ids = list(prefill_raw[row * row_width : (row + 1) * row_width])
        if decode_ids != prefill_ids:
            differences.append({"row": row, "decode": decode_ids, "prefill": prefill_ids})
    return {
        "first_predicting_row": 255,
        "last_predicting_row_exclusive": 511,
        "compared_rows": 256,
        "different_rows": differences,
        "different_row_count": len(differences),
    }


def _validate_prefill_rescore_comparison(
    value: Mapping[str, Any],
    input_ids: list[int],
    generated_ids: list[int],
    route: Mapping[str, Any],
) -> None:
    comparison = value.get("prefill_rescore_comparison")
    if not isinstance(comparison, Mapping):
        raise QualificationError("prefill-rescore diagnostic comparison is missing")
    for key in (
        "decode_request",
        "prefill_request",
        "rescore_request",
        "request_parameters",
        "response_sha256",
        "route_sha256",
    ):
        if not isinstance(comparison.get(key), Mapping):
            raise QualificationError(f"prefill-rescore comparison is missing {key}")
    decode_request = comparison["decode_request"]
    prefill_request = comparison["prefill_request"]
    if decode_request.get("token_ids") != input_ids[:256]:
        raise QualificationError("decode request prompt IDs do not align")
    if prefill_request.get("token_ids") != input_ids:
        raise QualificationError("prefill request token IDs do not align")
    if comparison["rescore_request"] != prefill_request:
        raise QualificationError("prefill-rescore request aliases differ")
    decode_salt = decode_request.get("cache_salt")
    prefill_salt = prefill_request.get("cache_salt")
    if (
        not isinstance(decode_salt, str)
        or not isinstance(prefill_salt, str)
        or not decode_salt
        or not prefill_salt
        or decode_salt == prefill_salt
    ):
        raise QualificationError("prefill-rescore cache salts are missing or not distinct")
    prefill_params = prefill_request.get("sampling_params")
    if not isinstance(prefill_params, Mapping):
        raise QualificationError("prefill-rescore sampling parameters are missing")
    if (
        value.get("cache_salt") != prefill_salt
        or value.get("rescore_cache_salt") != prefill_salt
        or value.get("decode_cache_salt") != decode_salt
        or value.get("sampling_params") != prefill_params
        or value.get("temperature") != prefill_params.get("temperature")
    ):
        raise QualificationError("prefill-rescore primary request fields do not match the reference")
    salts = comparison.get("salts")
    if (
        not isinstance(salts, Mapping)
        or salts.get("decode") != decode_salt
        or salts.get("prefill") != prefill_salt
        or salts.get("rescore") != prefill_salt
    ):
        raise QualificationError("prefill-rescore salt diagnostic differs")
    request_parameters = comparison["request_parameters"]
    if (
        request_parameters.get("decode") != decode_request.get("sampling_params")
        or request_parameters.get("prefill") != prefill_request.get("sampling_params")
    ):
        raise QualificationError("prefill-rescore request parameters differ")
    for hash_map_name in ("response_sha256", "route_sha256"):
        hashes = comparison[hash_map_name]
        for phase in ("decode", "prefill", "rescore"):
            digest = hashes.get(phase)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise QualificationError(f"prefill-rescore {hash_map_name} is invalid")
    responses = comparison.get("responses")
    if not isinstance(responses, Mapping):
        raise QualificationError("prefill-rescore exact responses are missing")
    for phase in ("decode", "prefill", "rescore"):
        if phase not in responses:
            raise QualificationError(f"prefill-rescore response is missing {phase}")
        if _json_sha256(responses[phase]) != comparison["response_sha256"][phase]:
            raise QualificationError(f"prefill-rescore {phase} response hash differs")
    if (
        responses["rescore"] != responses["prefill"]
        or comparison["response_sha256"].get("rescore")
        != comparison["response_sha256"].get("prefill")
    ):
        raise QualificationError("prefill-rescore response aliases differ")
    if (
        comparison["route_sha256"].get("prefill") != route.get("sha256")
        or comparison["route_sha256"].get("rescore") != route.get("sha256")
    ):
        raise QualificationError("prefill-rescore route hash does not match reference route")
    decode_route = comparison.get("decode_route")
    if not isinstance(decode_route, Mapping):
        raise QualificationError("prefill-rescore decode route is missing")
    _validate_route_payload(decode_route, 511, "decode")
    _validate_route_payload(route, 512, "prefill")
    if comparison["route_sha256"].get("decode") != decode_route.get("sha256"):
        raise QualificationError("prefill-rescore decode route hash does not match diagnostic route")
    if (
        value.get("response_sha256") != comparison["response_sha256"]
        or value.get("decode_route_sha256") != comparison["route_sha256"].get("decode")
        or value.get("rescore_route_sha256") != comparison["route_sha256"].get("prefill")
        or comparison["route_sha256"].get("rescore")
        != comparison["route_sha256"].get("prefill")
    ):
        raise QualificationError("prefill-rescore top-level hashes are not aliases")
    if (
        comparison.get("prefill_route") != route
        or comparison.get("rescore_route") != route
    ):
        raise QualificationError("prefill-rescore route aliases differ")
    deltas = comparison.get("completion_logprob_deltas")
    if (
        not isinstance(deltas, list)
        or len(deltas) != len(generated_ids)
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in deltas)
    ):
        raise QualificationError("prefill-rescore completion deltas are invalid")
    per_completion = comparison.get("per_completion")
    if not isinstance(per_completion, list) or len(per_completion) != len(generated_ids):
        raise QualificationError("prefill-rescore per-completion diagnostic is invalid")
    for index, entry in enumerate(per_completion):
        if (
            not isinstance(entry, Mapping)
            or entry.get("index") != index
            or entry.get("token_id") != generated_ids[index]
            or entry.get("delta") != deltas[index]
            or any(
                isinstance(entry.get(key), bool)
                or not isinstance(entry.get(key), (int, float))
                or not math.isfinite(float(entry.get(key)))
                for key in ("decode_logprob", "prefill_logprob", "delta")
            )
        ):
            raise QualificationError("prefill-rescore per-completion diagnostic is misaligned")
        expected_delta = float(entry["prefill_logprob"]) - float(entry["decode_logprob"])
        if not math.isclose(float(entry["delta"]), expected_delta, rel_tol=1e-12, abs_tol=1e-12):
            raise QualificationError("prefill-rescore completion delta was not recomputed")
    decode_values = _completion_evidence(responses["decode"], generated_ids)
    rescore_values = _prompt_evidence(responses["rescore"], input_ids)[256:]
    trainer_values = [
        _strict_logprob(logprob, "trainer inference logprob")
        for logprob in value["inference_logprobs"][256:]
    ]
    if any(
        float(entry["decode_logprob"]) != decoded
        or float(entry["prefill_logprob"]) != rescored
        or float(entry["prefill_logprob"]) != trainer
        for entry, decoded, rescored, trainer in zip(
            per_completion, decode_values, rescore_values, trainer_values
        )
    ):
        raise QualificationError("prefill-rescore completion evidence is not bound")
    route_differences = comparison.get("predicting_route_differences")
    if not isinstance(route_differences, Mapping):
        raise QualificationError("prefill-rescore route comparison is missing")
    if route_differences.get("compared_rows") != 256 or not isinstance(route_differences.get("different_rows"), list):
        raise QualificationError("prefill-rescore predicting routes are not aligned")
    recomputed_routes = _route_prediction_differences(decode_route, route)
    if route_differences != recomputed_routes:
        raise QualificationError("prefill-rescore predicting route differences are incorrect")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise QualificationError(f"cannot read qualification artifact {path}") from error
    return digest.hexdigest()


def validate_lora_config(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Require adapter settings to equal the trainer's configured settings."""
    if not isinstance(value, Mapping):
        raise QualificationError("adapter_config.json must be an object")
    for key in ("r", "lora_alpha", "lora_dropout"):
        expected_value = expected.get(key)
        actual_value = value.get(key)
        if key == "lora_dropout":
            try:
                actual_float = float(actual_value)
                expected_float = float(expected_value)
            except (TypeError, ValueError) as error:
                raise QualificationError(f"adapter config differs in {key}") from error
            if isinstance(actual_value, bool) or actual_float != expected_float:
                raise QualificationError(f"adapter config differs in {key}")
        elif type(actual_value) is not int or actual_value != expected_value:
            raise QualificationError(f"adapter config differs in {key}")
    expected_targets = expected.get("target_modules")
    actual_targets = value.get("target_modules")
    if (
        not isinstance(expected_targets, list)
        or not isinstance(actual_targets, list)
        or any(not isinstance(target, str) for target in expected_targets)
        or any(not isinstance(target, str) for target in actual_targets)
        or set(actual_targets) != set(expected_targets)
        or len(actual_targets) != len(expected_targets)
    ):
        raise QualificationError("adapter config target_modules differ")
    return dict(value)


def validate_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require one fresh, full-token, policy-0 trainer reference."""
    if not isinstance(value, Mapping):
        raise QualificationError("reference must be an object")
    if value.get("policy") != "policy-0":
        raise QualificationError("reference is not a policy-0 reference")
    request_mode = value.get("request_mode")
    if request_mode not in {
        "serialized_generate_completion_logprobs",
        _PREFILL_RESCORE_REQUEST_MODE,
    }:
        raise QualificationError("reference does not use completion qualification mode")
    if value.get("source_kind") in {"rollout_batch", "cached_rollout_batch"}:
        raise QualificationError("cached rollout reference is not allowed")
    input_ids = value.get("input_ids")
    logprobs = value.get("inference_logprobs")
    loss_mask = value.get("loss_mask")
    lengths = value.get("sequence_lengths")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(isinstance(token, bool) or not isinstance(token, int) for token in input_ids)
        or not isinstance(logprobs, list)
        or len(logprobs) != len(input_ids)
        or not isinstance(loss_mask, list)
        or len(loss_mask) != len(input_ids)
        or any(not isinstance(flag, bool) for flag in loss_mask)
        or not isinstance(lengths, list)
        or any(type(length) is not int or length <= 0 for length in lengths)
        or sum(lengths) != len(input_ids)
    ):
        raise QualificationError("reference token, logprob, mask, or length fields do not align")
    first_generated = loss_mask.index(True) if any(loss_mask) else len(loss_mask)
    expected_source_hash = hashlib.sha256(
        json.dumps(input_ids[:first_generated], separators=(",", ":")).encode()
    ).hexdigest()
    if value.get("source_input_sha256") != expected_source_hash:
        raise QualificationError("reference source input hash differs")
    if not any(loss_mask) or all(loss_mask):
        raise QualificationError("reference must contain prompt and generated loss-mask regions")
    first_generated = loss_mask.index(True)
    if any(loss_mask[:first_generated]) or not all(loss_mask[first_generated:]):
        raise QualificationError("reference loss mask is not prompt-then-completion")
    route = value.get("routed_experts")
    expected_route_rows = len(input_ids) if request_mode == _PREFILL_RESCORE_REQUEST_MODE else len(input_ids) - 1
    if not isinstance(route, Mapping) or route.get("shape", [None])[0] != expected_route_rows:
        raise QualificationError("reference routes have an invalid row count")
    required_fields = ["policy_identity_before", "policy_identity_after", "cache_salt"]
    if request_mode == _PREFILL_RESCORE_REQUEST_MODE:
        required_fields.extend(("policy_identity_between", "rescore_cache_salt", "decode_cache_salt"))
    for field in required_fields:
        if field not in value or value[field] is None:
            raise QualificationError(f"reference is missing {field}")
    if value["policy_identity_before"] != value["policy_identity_after"]:
        raise QualificationError("reference policy identity changed during generation")
    generated_ids = value.get("generated_token_ids")
    if (
        not isinstance(generated_ids, list)
        or any(isinstance(token, bool) or not isinstance(token, int) for token in generated_ids)
        or value.get("generated_token_count") != len(generated_ids)
        or len(generated_ids) != sum(loss_mask)
    ):
        raise QualificationError("reference generated token metadata is not aligned")
    if any(logprob != 0.0 for logprob, trainable in zip(logprobs, loss_mask) if not trainable):
        raise QualificationError("reference prompt inference logprobs must be zero")
    if any(
        isinstance(logprob, bool)
        or not isinstance(logprob, (int, float))
        or not math.isfinite(float(logprob))
        for logprob in logprobs
    ):
        raise QualificationError("reference inference logprobs must be finite")
    if request_mode == _PREFILL_RESCORE_REQUEST_MODE:
        if len(input_ids) != 512 or sum(loss_mask) != 256 or loss_mask != [False] * 256 + [True] * 256:
            raise QualificationError("prefill-rescore reference token alignment is invalid")
        if value["policy_identity_between"] != value["policy_identity_before"]:
            raise QualificationError("policy receipt changed between decode and prefill")
        if value.get("decode_cache_salt") == value.get("cache_salt"):
            raise QualificationError("prefill-rescore cache salts must be distinct")
        if generated_ids != input_ids[256:]:
            raise QualificationError("prefill-rescore completion suffix does not align")
        _validate_prefill_rescore_comparison(value, input_ids, generated_ids, route)
    return dict(value)


def validate_local_artifacts(
    adapter_path: Path,
    config_path: Path,
    reference_path: Path,
    trainer_lora: Mapping[str, Any],
) -> dict[str, dict[str, int | str]]:
    """Validate the source files and return exact identities for each file."""
    try:
        config = json.loads(config_path.read_text())
        reference = json.loads(reference_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError("qualification source JSON is unreadable") from error
    validate_lora_config(config, trainer_lora)
    validate_reference(reference)
    files = {
        "adapter_model.safetensors": adapter_path,
        "adapter_config.json": config_path,
        "reference.json": reference_path,
    }
    result: dict[str, dict[str, int | str]] = {}
    for name, path in files.items():
        if not path.is_file() or path.is_symlink():
            raise QualificationError(f"qualification source file is missing: {path}")
        result[name] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return result
