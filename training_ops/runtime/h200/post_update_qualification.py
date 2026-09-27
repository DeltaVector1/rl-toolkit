"""Fail-closed checks for the separate H200 post-update smoke qualification.

This module does not start a trainer or inference service.  It validates the
local evidence that an operator must collect after one real trainer update:
the reference, exported adapter, transfer, load, and post-update receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import base64
import binascii
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

from .prepare_trinity_policy0_qualification import (
    QualificationError,
    validate_reference as validate_policy0_reference,
)


POST_UPDATE_STEP = 1
POST_UPDATE_LOAD_SLOT = "trinity-h200-post-update-qualification-slot1"
POST_UPDATE_REQUEST_MODE = "serialized_generate_completion_logprobs"
POST_UPDATE_SEQ_LEN = 8192
POST_UPDATE_PROMPT_TOKENS = 8160
POST_UPDATE_COMPLETION_TOKENS = 32
POST_UPDATE_ROUTE_SHAPE = [8191, 60, 4]
POST_UPDATE_ADVANTAGE_VALUE = 1.0
FORWARD_BACKWARD_GATE_SECONDS = 600.0
_POST_UPDATE_SAMPLING_PARAMS = {
    "max_tokens": POST_UPDATE_COMPLETION_TOKENS,
    "temperature": 0.0,
    "top_p": 1.0,
    "logprobs": 1,
    "min_tokens": POST_UPDATE_COMPLETION_TOKENS,
}

_EXPECTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert",
    "experts",
}
_IDENTITY_FIELDS = ("target_sha256", "target_size", "config_sha256", "config_size")
_RECEIPT_FIELDS = (
    "source_url",
    "target_sha256",
    "target_size",
    "config_sha256",
    "config_size",
    "lora_name",
    "server_instance",
)


class PostUpdateQualificationError(ValueError):
    """The post-update smoke qualification evidence is unsafe or incomplete."""


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_no_clobber_write(path: Path, payload: str) -> None:
    """Publish one final artifact without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise PostUpdateQualificationError(
                f"refusing to clobber existing qualification output: {path}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_post_failure(
    output_path: Path,
    *,
    error: str,
    request: Mapping[str, Any],
    response: Any = None,
    client_request_id: str | None = None,
) -> None:
    """Keep the first failed POST's raw evidence without overwriting it."""
    diagnostic: dict[str, Any] = {
        "schema_version": 1,
        "kind": "post_update_post_failure",
        "error": error,
        "request": dict(request),
        "request_sha256": _json_sha256(request),
        "client_request_id": client_request_id,
        "response": response if isinstance(response, Mapping) else None,
    }
    if isinstance(response, Mapping):
        diagnostic["response_sha256"] = _json_sha256(response)
        diagnostic["server_response_id"] = response.get("request_id")
    else:
        diagnostic["response_repr"] = repr(response)
    _atomic_no_clobber_write(
        output_path,
        json.dumps(diagnostic, sort_keys=True, separators=(",", ":")) + "\n",
    )


def _post_update_binding(
    reference: Mapping[str, Any],
    *,
    advantage_value: float = POST_UPDATE_ADVANTAGE_VALUE,
) -> dict[str, Any]:
    receipt = reference.get("policy_identity_before")
    request_bodies = reference.get("request_bodies")
    responses = reference.get("responses")
    route = reference.get("routed_experts")
    if not isinstance(receipt, Mapping) or not isinstance(request_bodies, Mapping) or not isinstance(responses, Mapping):
        raise PostUpdateQualificationError("post-update reference request or receipt evidence is missing")
    if not isinstance(route, Mapping) or route.get("shape") != POST_UPDATE_ROUTE_SHAPE:
        raise PostUpdateQualificationError("post-update reference route shape is invalid")
    request = request_bodies.get("decode")
    response = responses.get("decode")
    if not isinstance(request, Mapping) or not isinstance(response, Mapping):
        raise PostUpdateQualificationError("post-update reference decode evidence is missing")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise PostUpdateQualificationError("post-update request ID is missing")
    request_without_id = dict(request)
    request_without_id.pop("request_id", None)
    if request_id != _json_sha256(request_without_id):
        raise PostUpdateQualificationError("post-update client request ID is not bound to the request body")
    response_id = response.get("request_id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise PostUpdateQualificationError("post-update response ID is missing")
    server_instance = receipt.get("server_instance")
    target_sha = receipt.get("target_sha256")
    config_sha = receipt.get("config_sha256")
    if not isinstance(server_instance, str) or not server_instance:
        raise PostUpdateQualificationError("post-update reference server identity is missing")
    if not _valid_digest(target_sha) or not _valid_digest(config_sha):
        raise PostUpdateQualificationError("post-update reference adapter identity is invalid")
    input_ids = reference.get("input_ids")
    loss_mask = reference.get("loss_mask")
    if not isinstance(input_ids, list) or not isinstance(loss_mask, list):
        raise PostUpdateQualificationError("post-update reference input fields are missing")
    first_generated = loss_mask.index(True) if True in loss_mask else len(loss_mask)
    return {
        "seq_len": POST_UPDATE_SEQ_LEN,
        "prompt_tokens": POST_UPDATE_PROMPT_TOKENS,
        "completion_tokens": POST_UPDATE_COMPLETION_TOKENS,
        "mask_false": POST_UPDATE_PROMPT_TOKENS,
        "mask_true": POST_UPDATE_COMPLETION_TOKENS,
        "request_mode": POST_UPDATE_REQUEST_MODE,
        "policy": reference.get("policy"),
        "policy_model": reference.get("model"),
        "policy_receipt": dict(receipt),
        "receipt_sha256": _json_sha256(receipt),
        "adapter_sha256": target_sha,
        "config_sha256": config_sha,
        "server_instance": server_instance,
        "server_instance_sha256": hashlib.sha256(server_instance.encode()).hexdigest(),
        "input_sha256": hashlib.sha256(
            json.dumps(input_ids[:first_generated], separators=(",", ":")).encode()
        ).hexdigest(),
        "full_input_sha256": _json_sha256(input_ids),
        "request_sha256": _json_sha256(request),
        "response_sha256": _json_sha256(response),
        "client_request_id": request_id,
        "server_response_id": response.get("request_id"),
        "adapter_size": receipt.get("target_size"),
        "config_size": receipt.get("config_size"),
        "route_shape": list(route["shape"]),
        "route_sha256": route.get("sha256"),
        "advantage_value": advantage_value,
    }


def build_post_update_reference(
    *,
    endpoint: str,
    input_record: Path | Mapping[str, Any],
    output_path: Path,
    base_model: str,
    policy_model: str,
    expected_policy_identity: Path | Mapping[str, Any],
    policy_receipt: Path | Mapping[str, Any],
    cache_salt: str,
    readiness_endpoint: str,
    policy: str = "policy-0",
    timeout: float = 120.0,
    post_json: Any = None,
    get_json: Any = None,
    advantage_value: float = POST_UPDATE_ADVANTAGE_VALUE,
    failure_output_path: Path | None = None,
) -> dict[str, Any]:
    """Build one strict reference from one fresh serialized completion decode."""
    if policy not in {"policy-0", "policy-1"}:
        raise PostUpdateQualificationError(
            "post-update reference policy must be policy-0 or policy-1"
        )
    failure_path = Path(failure_output_path) if failure_output_path is not None else output_path
    if (
        output_path.exists()
        or output_path.is_symlink()
        or failure_path.exists()
        or failure_path.is_symlink()
    ):
        raise PostUpdateQualificationError(
            f"refusing to clobber existing qualification output: {failure_path}"
        )
    if isinstance(advantage_value, bool) or not math.isfinite(float(advantage_value)) or float(advantage_value) == 0.0:
        raise PostUpdateQualificationError("post-update advantage value must be finite and nonzero")
    from . import build_trinity_parity_reference as reference_builder
    from .build_trinity_parity_reference import build_reference

    kwargs: dict[str, Any] = {}
    if get_json is not None:
        kwargs["get_json"] = get_json
    actual_post_json = post_json or reference_builder._post_json

    def request_with_identity(url: str, body: Mapping[str, Any], request_timeout: float) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise PostUpdateQualificationError("post-update request body is not mutable")
        request_id = _json_sha256(body)
        body["request_id"] = request_id
        request_snapshot = dict(body)
        try:
            response = actual_post_json(url, body, request_timeout)
        except Exception as error:
            _publish_post_failure(
                failure_path,
                error=str(error),
                request=request_snapshot,
                client_request_id=request_id,
            )
            raise PostUpdateQualificationError(
                "post-update POST failed; raw request diagnostic was published"
            ) from error
        if not isinstance(response, Mapping):
            error = "post-update response is not an object"
            _publish_post_failure(
                failure_path,
                error=error,
                request=request_snapshot,
                response=response,
                client_request_id=request_id,
            )
            raise PostUpdateQualificationError(error)
        returned_id = response.get("request_id")
        if not isinstance(returned_id, str) or not returned_id.strip():
            error = "post-update response request ID is missing"
            _publish_post_failure(
                failure_path,
                error=error,
                request=request_snapshot,
                response=response,
                client_request_id=request_id,
            )
            raise PostUpdateQualificationError(error)
        return dict(response)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=f".{output_path.name}.build.") as directory:
        temporary_output = Path(directory) / output_path.name
        try:
            result = build_reference(
                endpoint=endpoint,
                input_record=input_record,
                output_path=temporary_output,
                base_model=base_model,
                policy=policy,
                policy_model=policy_model,
                policy_receipt=policy_receipt,
                expected_policy_identity=expected_policy_identity,
                cache_salt=cache_salt,
                prefill_rescore=False,
                readiness_endpoint=readiness_endpoint,
                completion_tokens=POST_UPDATE_COMPLETION_TOKENS,
                minimum_completion_tokens=POST_UPDATE_COMPLETION_TOKENS,
                trainer_seq_len=POST_UPDATE_SEQ_LEN,
                minimum_tokens=POST_UPDATE_PROMPT_TOKENS,
                expected_route_shape=POST_UPDATE_ROUTE_SHAPE,
                timeout=timeout,
                post_json=request_with_identity,
                **kwargs,
            )
        except Exception as error:
            if error.__class__.__name__ == "ReferenceError":
                raise PostUpdateQualificationError(str(error)) from error
            raise
        if result.get("request_mode") != POST_UPDATE_REQUEST_MODE or "prefill_rescore_comparison" in result:
            raise PostUpdateQualificationError("post-update reference must use one completion decode")
        if len(result.get("input_ids", [])) != POST_UPDATE_SEQ_LEN:
            raise PostUpdateQualificationError("post-update reference length is not 8192")
        result["advantages"] = [0.0] * POST_UPDATE_PROMPT_TOKENS + [float(advantage_value)] * POST_UPDATE_COMPLETION_TOKENS
        result["client_request_id"] = result["request_bodies"]["decode"]["request_id"]
        result["server_response_id"] = result["responses"]["decode"]["request_id"]
        result["qualification_binding"] = _post_update_binding(
            result, advantage_value=float(advantage_value)
        )
        # The builder's temporary file is discarded; only this final link is public.
        temporary_output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
        _atomic_no_clobber_write(output_path, temporary_output.read_text(encoding="utf-8"))
    return result


def build_post_update_references(
    *,
    input_records: list[Path | Mapping[str, Any]],
    output_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build exactly eight distinct decode references for PP8 fake data."""
    if len(input_records) != 8:
        raise PostUpdateQualificationError("post-update qualification requires eight distinct references")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    post_json = kwargs.pop("post_json", None)
    get_json = kwargs.pop("get_json", None)
    cache_salt = kwargs.pop("cache_salt", None)
    if not isinstance(cache_salt, str) or not cache_salt:
        raise PostUpdateQualificationError("post-update qualification requires a fresh cache salt")
    if output_path.exists() or output_path.is_symlink():
        raise PostUpdateQualificationError(
            f"refusing to clobber existing qualification output: {output_path}"
        )
    prompt_hashes: list[str] = []
    for record in input_records:
        if isinstance(record, Path):
            try:
                loaded = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise PostUpdateQualificationError("post-update input record is invalid") from error
        else:
            loaded = record
        if not isinstance(loaded, Mapping) or not isinstance(loaded.get("input_ids"), list):
            raise PostUpdateQualificationError("post-update input record must contain input_ids")
        if len(loaded["input_ids"]) != POST_UPDATE_PROMPT_TOKENS:
            raise PostUpdateQualificationError("post-update input records must contain 8160 prompt tokens")
        prompt_hashes.append(_json_sha256(loaded["input_ids"]))
    if len(set(prompt_hashes)) != 8:
        raise PostUpdateQualificationError("post-update input records must be eight distinct prompts")
    built: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=output_path.parent) as directory:
        for index, record in enumerate(input_records):
            per_reference = dict(kwargs)
            per_reference["cache_salt"] = f"{cache_salt}-{index}"
            current = build_post_update_reference(
                input_record=record,
                output_path=Path(directory) / f"reference-{index}.json",
                failure_output_path=output_path,
                post_json=post_json,
                get_json=get_json,
                **per_reference,
            )
            built.append(current)
    input_hashes = [item["qualification_binding"]["input_sha256"] for item in built]
    response_hashes = [item["qualification_binding"]["response_sha256"] for item in built]
    if len(set(input_hashes)) != 8 or len(set(response_hashes)) != 8:
        raise PostUpdateQualificationError("post-update references must have unique input and response hashes")
    result = {"schema_version": 1, "microbatches": built}
    _atomic_no_clobber_write(output_path, json.dumps(result, separators=(",", ":")) + "\n")
    return result


def post_update_adapter_path(output_dir: str | Path) -> Path:
    """Return the only exported step path accepted by this qualification."""
    root = Path(output_dir)
    return root / "run_trainer_only" / "broadcasts" / f"step_{POST_UPDATE_STEP}"


def post_update_load_target(output_dir: str | Path) -> dict[str, Any]:
    """Return the verified step and slot used to load the real update."""
    return {
        "step": POST_UPDATE_STEP,
        "slot": POST_UPDATE_LOAD_SLOT,
        "adapter_path": str(post_update_adapter_path(output_dir)),
    }


def validate_forward_backward_seconds(value: Any) -> float:
    """Keep the existing 600-second F+B gate separate from adapter timings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostUpdateQualificationError("forward/backward time is not numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise PostUpdateQualificationError("forward/backward time is not finite")
    if value > FORWARD_BACKWARD_GATE_SECONDS:
        raise PostUpdateQualificationError(
            f"forward/backward time exceeds the {FORWARD_BACKWARD_GATE_SECONDS:g}s gate"
        )
    return value


def _validate_post_update_route(value: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "encoding",
        "dtype",
        "shape",
        "dense_layers",
        "experts",
        "sha256",
        "data",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PostUpdateQualificationError("post-update reference route schema is invalid")
    shape = value.get("shape")
    if shape != POST_UPDATE_ROUTE_SHAPE or value.get("schema_version") != 1 or value.get("encoding") != "base64" or value.get("dtype") != "uint8" or value.get("dense_layers") != 6 or value.get("experts") != 256:
        raise PostUpdateQualificationError("post-update reference route metadata is invalid")
    encoded = value.get("data")
    digest = value.get("sha256")
    if not isinstance(encoded, str) or not _valid_digest(digest):
        raise PostUpdateQualificationError("post-update reference route checksum metadata is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise PostUpdateQualificationError("post-update reference route data is invalid") from error
    if len(raw) != shape[0] * shape[1] * shape[2] or hashlib.sha256(raw).hexdigest() != digest:
        raise PostUpdateQualificationError("post-update reference route checksum differs")
    row_width = shape[1] * shape[2]
    for offset in range(0, len(raw), row_width):
        row = raw[offset : offset + row_width]
        if any(row[: 6 * shape[2]]):
            raise PostUpdateQualificationError("post-update reference dense route IDs are invalid")
        for layer in range(6, shape[1]):
            selected = row[layer * shape[2] : (layer + 1) * shape[2]]
            if len(set(selected)) != len(selected) or any(item >= 256 for item in selected):
                raise PostUpdateQualificationError("post-update reference sparse route IDs are invalid")


def validate_post_update_reference(
    value: Mapping[str, Any],
    *,
    expected_seq_len: int | None = None,
    expected_policy_identity: Mapping[str, Any] | None = None,
    expected_policy: str = "policy-0",
    expected_policy_model: str | None = None,
    expected_response: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one non-rescore reference with real, aligned advantages."""
    if not isinstance(value, Mapping):
        raise PostUpdateQualificationError("post-update reference must be an object")
    request_mode = value.get("request_mode")
    if request_mode == "serialized_generate_decode256_prefill_rescore":
        raise PostUpdateQualificationError(
            "post-update qualification rejects rescore-mode references"
        )
    if request_mode != POST_UPDATE_REQUEST_MODE:
        raise PostUpdateQualificationError(
            "post-update reference must use completion qualification mode"
        )
    try:
        validated = validate_policy0_reference(
            value,
            expected_policy=expected_policy,
        )
    except QualificationError as error:
        raise PostUpdateQualificationError(str(error)) from error

    input_ids = validated["input_ids"]
    loss_mask = validated["loss_mask"]
    advantages = validated.get("advantages")
    if not isinstance(advantages, list):
        raise PostUpdateQualificationError(
            "post-update reference advantages are missing"
        )
    if len(advantages) != len(input_ids):
        raise PostUpdateQualificationError(
            "post-update reference advantages do not align with input tokens"
        )
    numeric: list[float] = []
    for index, advantage in enumerate(advantages):
        if isinstance(advantage, bool) or not isinstance(advantage, (int, float)):
            raise PostUpdateQualificationError(
                f"post-update reference advantage {index} is not numeric"
            )
        value_float = float(advantage)
        if not math.isfinite(value_float):
            raise PostUpdateQualificationError(
                f"post-update reference advantage {index} is not finite"
            )
        if not loss_mask[index] and value_float != 0.0:
            raise PostUpdateQualificationError(
                "post-update reference advantages are misaligned with the loss mask"
            )
        numeric.append(value_float)
    if not any(value_float != 0.0 for value_float, keep in zip(numeric, loss_mask) if keep):
        raise PostUpdateQualificationError(
            "post-update reference advantages are all zero"
        )
    if expected_seq_len is None:
        return dict(validated)
    if expected_seq_len != POST_UPDATE_SEQ_LEN:
        raise PostUpdateQualificationError("post-update config seq_len must be 8192")
    if len(input_ids) != POST_UPDATE_SEQ_LEN or validated.get("sequence_lengths") != [POST_UPDATE_SEQ_LEN]:
        raise PostUpdateQualificationError("post-update reference length does not match config seq_len=8192")
    if loss_mask != [False] * POST_UPDATE_PROMPT_TOKENS + [True] * POST_UPDATE_COMPLETION_TOKENS:
        raise PostUpdateQualificationError("post-update reference loss mask must be 8160 false and 32 true")
    if validated.get("routed_experts", {}).get("shape") != POST_UPDATE_ROUTE_SHAPE:
        raise PostUpdateQualificationError("post-update reference route shape must be [8191,60,4]")
    _validate_post_update_route(validated["routed_experts"])
    request_bodies = validated.get("request_bodies")
    responses = validated.get("responses")
    stored_request_hashes = validated.get("request_sha256")
    stored_response_hashes = validated.get("response_sha256")
    if (
        not isinstance(request_bodies, Mapping)
        or not isinstance(responses, Mapping)
        or not isinstance(stored_request_hashes, Mapping)
        or not isinstance(stored_response_hashes, Mapping)
        or not isinstance(request_bodies.get("decode"), Mapping)
        or not isinstance(responses.get("decode"), Mapping)
        or stored_request_hashes.get("decode") != _json_sha256(request_bodies["decode"])
        or stored_response_hashes.get("decode") != _json_sha256(responses["decode"])
    ):
        raise PostUpdateQualificationError("post-update request or response hash differs")
    request = request_bodies["decode"]
    if request.get("sampling_params") != _POST_UPDATE_SAMPLING_PARAMS:
        raise PostUpdateQualificationError("post-update sampling parameters are not deterministic")
    if request.get("token_ids") != input_ids[:POST_UPDATE_PROMPT_TOKENS]:
        raise PostUpdateQualificationError("post-update request tokens differ from the reference prompt")
    if request.get("cache_salt") != validated.get("cache_salt"):
        raise PostUpdateQualificationError("post-update request cache salt differs")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise PostUpdateQualificationError("post-update request ID is missing")
    request_without_id = dict(request)
    request_without_id.pop("request_id", None)
    if request_id != _json_sha256(request_without_id):
        raise PostUpdateQualificationError("post-update client request ID is not bound to the request body")
    response = responses["decode"]
    response_id = response.get("request_id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise PostUpdateQualificationError("post-update response ID is missing")
    if (
        validated.get("client_request_id") != request_id
        or validated.get("server_response_id") != response_id
        or validated.get("client_request_id") != request.get("request_id")
        or validated.get("server_response_id") != response.get("request_id")
    ):
        raise PostUpdateQualificationError("post-update request or response ID binding differs")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise PostUpdateQualificationError("post-update response choice is missing")
    choice = choices[0]
    if choice.get("token_ids") != validated.get("generated_token_ids"):
        raise PostUpdateQualificationError("post-update response token IDs differ")
    try:
        from .build_trinity_parity_reference import _flatten_completion_logprobs, _normalise_route

        response_logprobs = _flatten_completion_logprobs(
            choice.get("logprobs"), list(validated.get("generated_token_ids", []))
        )
        response_route = _normalise_route(choice.get("routed_experts"), POST_UPDATE_SEQ_LEN - 1)
    except Exception as error:
        raise PostUpdateQualificationError("post-update response token evidence is invalid") from error
    if response_logprobs != validated["inference_logprobs"][POST_UPDATE_PROMPT_TOKENS:]:
        raise PostUpdateQualificationError("post-update response logprobs differ")
    if response_route != validated.get("routed_experts"):
        raise PostUpdateQualificationError("post-update response route differs")
    before = validated.get("policy_identity_before")
    after = validated.get("policy_identity_after")
    policy = validated.get("policy_identity")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping) or before != after or policy != before:
        raise PostUpdateQualificationError("post-update policy receipt is stale or mismatched")
    if expected_policy_model is not None and before.get("lora_name") != expected_policy_model:
        raise PostUpdateQualificationError("post-update policy receipt is not bound to the expected loaded slot")
    if expected_policy_identity is not None and dict(before) != dict(expected_policy_identity):
        raise PostUpdateQualificationError("post-update policy receipt differs from expected receipt")
    advantages = validated.get("advantages")
    if not isinstance(advantages, list) or len(advantages) != POST_UPDATE_SEQ_LEN:
        raise PostUpdateQualificationError("post-update reference advantages do not align with seq_len")
    for index, advantage in enumerate(advantages):
        if isinstance(advantage, bool) or not isinstance(advantage, (int, float)) or not math.isfinite(float(advantage)):
            raise PostUpdateQualificationError(f"post-update reference advantage {index} is not finite")
        expected = 0.0 if not loss_mask[index] else POST_UPDATE_ADVANTAGE_VALUE
        if float(advantage) != expected:
            raise PostUpdateQualificationError("post-update reference advantages are not deterministically aligned")
    expected_binding = _post_update_binding(validated)
    binding = validated.get("qualification_binding")
    if binding != expected_binding:
        raise PostUpdateQualificationError("post-update reference receipt or hash binding differs")
    if expected_response is not None and _json_sha256(expected_response) != binding["response_sha256"]:
        raise PostUpdateQualificationError("post-update reference response hash differs")
    return dict(validated)


def validate_post_update_references(
    value: Mapping[str, Any],
    *,
    expected_seq_len: int = POST_UPDATE_SEQ_LEN,
    expected_policy_identity: Mapping[str, Any] | None = None,
    expected_policy: str = "policy-0",
    expected_policy_model: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise PostUpdateQualificationError("post-update reference collection schema is invalid")
    microbatches = value.get("microbatches")
    if not isinstance(microbatches, list) or len(microbatches) != 8:
        raise PostUpdateQualificationError("post-update qualification requires eight microbatches")
    validated = [
        validate_post_update_reference(
            item,
            expected_seq_len=expected_seq_len,
            expected_policy_identity=expected_policy_identity,
            expected_policy=expected_policy,
            expected_policy_model=expected_policy_model,
        )
        for item in microbatches
    ]
    input_hashes = [item["qualification_binding"]["input_sha256"] for item in validated]
    response_hashes = [item["qualification_binding"]["response_sha256"] for item in validated]
    salts = [item.get("cache_salt") for item in validated]
    client_request_ids = [item["qualification_binding"]["client_request_id"] for item in validated]
    server_response_ids = [item["qualification_binding"]["server_response_id"] for item in validated]
    if (
        len(set(input_hashes)) != 8
        or len(set(response_hashes)) != 8
        or len(set(salts)) != 8
        or any(not isinstance(salt, str) or not salt for salt in salts)
        or len(set(client_request_ids)) != 8
        or len(set(server_response_ids)) != 8
    ):
        raise PostUpdateQualificationError("post-update microbatch input or response hashes are reused")
    return {"schema_version": 1, "microbatches": validated}


def _regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PostUpdateQualificationError(f"{label} is missing or not regular: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PostUpdateQualificationError(f"cannot read adapter artifact: {path}") from error
    return digest.hexdigest()


def _all_zero(path: Path) -> bool:
    with path.open("rb") as stream:
        saw_bytes = False
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            saw_bytes = saw_bytes or bool(block)
            if any(block):
                return False
    return not saw_bytes


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        for left_block, right_block in zip(
            iter(lambda: left_stream.read(1024 * 1024), b""),
            iter(lambda: right_stream.read(1024 * 1024), b""),
        ):
            if left_block != right_block:
                return False
    return True


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _adapter_identity(adapter_dir: Path) -> dict[str, Any]:
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise PostUpdateQualificationError("post-update adapter directory is missing")
    if adapter_dir.name != f"step_{POST_UPDATE_STEP}" or adapter_dir.parent.name != "broadcasts" or adapter_dir.parent.parent.name != "run_trainer_only":
        raise PostUpdateQualificationError("post-update adapter path is not the verified step-1 slot")
    adapter = adapter_dir / "adapter_model.safetensors"
    config = adapter_dir / "adapter_config.json"
    sidecar = adapter_dir / "adapter_model.safetensors.sha256"
    manifest_path = adapter_dir / "adapter_manifest.json"
    stable = adapter_dir / "STABLE"
    for path, label in (
        (adapter, "post-update adapter weights"),
        (config, "post-update adapter config"),
        (sidecar, "post-update adapter checksum"),
        (manifest_path, "post-update adapter manifest"),
        (stable, "post-update adapter STABLE marker"),
    ):
        _regular(path, label)
    target_sha = _sha256(adapter)
    config_sha = _sha256(config)
    target_size = adapter.stat().st_size
    config_size = config.stat().st_size
    if target_size <= 0 or _all_zero(adapter):
        raise PostUpdateQualificationError("post-update adapter weights are empty or all zero")
    try:
        sidecar_value = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise PostUpdateQualificationError("post-update adapter checksum is invalid") from error
    if sidecar_value != target_sha:
        raise PostUpdateQualificationError("post-update adapter checksum sidecar differs")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        adapter_config = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise PostUpdateQualificationError("post-update adapter metadata is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("kind") != "lora-adapter"
        or manifest.get("mode") != "full"
        or manifest.get("target_step") != POST_UPDATE_STEP
        or manifest.get("base_step") is not None
        or manifest.get("target_sha256") != target_sha
        or manifest.get("target_size") != target_size
        or manifest.get("full_sha256") != target_sha
        or manifest.get("full_size") != target_size
        or manifest.get("config_sha256") != config_sha
        or manifest.get("config_size") != config_size
    ):
        raise PostUpdateQualificationError("post-update adapter manifest identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise PostUpdateQualificationError("post-update adapter manifest file list is invalid")
    records = {item.get("name"): item for item in files if isinstance(item, dict)}
    if set(records) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise PostUpdateQualificationError("post-update adapter manifest file list is invalid")
    for name, size, digest in (
        ("adapter_config.json", config_size, config_sha),
        ("adapter_model.safetensors", target_size, target_sha),
    ):
        record = records[name]
        if record.get("size") != size or record.get("sha256") != digest:
            raise PostUpdateQualificationError("post-update adapter manifest file identity differs")
    if (
        type(adapter_config.get("r")) is not int
        or adapter_config.get("r") != 32
        or type(adapter_config.get("lora_alpha")) is not int
        or adapter_config.get("lora_alpha") != 64
        or isinstance(adapter_config.get("lora_dropout"), bool)
        or not isinstance(adapter_config.get("lora_dropout"), (int, float))
        or not math.isfinite(float(adapter_config.get("lora_dropout")))
        or float(adapter_config.get("lora_dropout")) != 0.0
        or not isinstance(adapter_config.get("target_modules"), list)
        or any(not isinstance(target, str) for target in adapter_config.get("target_modules", []))
        or set(adapter_config["target_modules"]) != _EXPECTED_TARGETS
        or len(adapter_config["target_modules"]) != len(_EXPECTED_TARGETS)
    ):
        raise PostUpdateQualificationError("post-update adapter LoRA configuration differs")
    expected_entries = {
        "STABLE",
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.safetensors.sha256",
        "adapter_manifest.json",
    }
    if {path.name for path in adapter_dir.iterdir()} != expected_entries:
        raise PostUpdateQualificationError("post-update adapter has unexpected files")
    return {
        "adapter_path": adapter,
        "config_path": config,
        "manifest_path": manifest_path,
        "stable_path": stable,
        "target_sha256": target_sha,
        "target_size": target_size,
        "config_sha256": config_sha,
        "config_size": config_size,
        "manifest_sha256": _sha256(manifest_path),
    }


def _check_identity(value: Mapping[str, Any], identity: Mapping[str, Any], label: str) -> None:
    for field in _IDENTITY_FIELDS:
        if field.endswith("_sha256") and not _valid_digest(value.get(field)):
            raise PostUpdateQualificationError(f"{label} {field} is invalid")
        if field.endswith("_size") and (type(value.get(field)) is not int or value[field] < 0):
            raise PostUpdateQualificationError(f"{label} {field} is invalid")
        if value.get(field) != identity[field]:
            raise PostUpdateQualificationError(f"{label} {field} differs from exported adapter")


def _check_receipt(value: Mapping[str, Any], identity: Mapping[str, Any], label: str, source_url: str | None = None) -> str:
    if not isinstance(value, Mapping):
        raise PostUpdateQualificationError(f"{label} evidence is missing")
    for field in _RECEIPT_FIELDS:
        if field not in value:
            raise PostUpdateQualificationError(f"{label} receipt is missing {field}")
    if not isinstance(value["source_url"], str) or not value["source_url"]:
        raise PostUpdateQualificationError(f"{label} receipt source URL is invalid")
    if source_url is not None and value["source_url"] != source_url:
        raise PostUpdateQualificationError(f"{label} receipt source URL differs")
    if value["lora_name"] != POST_UPDATE_LOAD_SLOT or not isinstance(value["server_instance"], str) or not value["server_instance"]:
        raise PostUpdateQualificationError(f"{label} receipt slot or server identity is invalid")
    _check_identity(value, identity, label)
    return value["source_url"]


def validate_post_update_evidence(
    adapter_dir: Path,
    policy0_adapter_path: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a fresh exported step, transfer, load, and post-update receipt."""
    if not isinstance(evidence, Mapping):
        raise PostUpdateQualificationError("post-update evidence is missing")
    for field in ("save", "transfer", "load", "post_update"):
        if not isinstance(evidence.get(field), Mapping):
            raise PostUpdateQualificationError(f"post-update {field} evidence is missing")
    policy0_adapter_path = Path(policy0_adapter_path)
    _regular(policy0_adapter_path, "policy-0 adapter")
    identity = _adapter_identity(Path(adapter_dir))
    policy0_sha = _sha256(policy0_adapter_path)
    if identity["target_sha256"] == policy0_sha or _same_bytes(identity["adapter_path"], policy0_adapter_path):
        raise PostUpdateQualificationError(
            "post-update adapter bytes or SHA differs from policy-0"
        )

    save = evidence["save"]
    if save.get("path") != str(identity["adapter_path"]) or save.get("step") != POST_UPDATE_STEP:
        raise PostUpdateQualificationError("post-update save evidence does not name exported step 1")
    _check_identity(save, identity, "post-update save evidence")
    if save.get("manifest_sha256") != identity["manifest_sha256"]:
        raise PostUpdateQualificationError("post-update save evidence manifest differs")

    transfer = evidence["transfer"]
    if transfer.get("path") != str(Path(adapter_dir)):
        raise PostUpdateQualificationError("post-update transfer evidence path differs")
    _check_identity(transfer, identity, "post-update transfer evidence")

    source_url = _check_receipt(evidence["load"], identity, "post-update load")
    post_update = evidence["post_update"]
    if post_update.get("step") != POST_UPDATE_STEP:
        raise PostUpdateQualificationError("post-update receipt does not name exported step 1")
    _check_receipt(post_update, identity, "post-update", source_url)
    return {
        "step": POST_UPDATE_STEP,
        "slot": POST_UPDATE_LOAD_SLOT,
        "adapter_path": str(identity["adapter_path"]),
        "source_url": source_url,
        "target_sha256": identity["target_sha256"],
        "target_size": identity["target_size"],
        "config_sha256": identity["config_sha256"],
        "config_size": identity["config_size"],
    }


def _post_update_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--input-record", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument(
        "--policy",
        required=True,
        choices=("policy-0", "policy-1"),
        help="loaded policy identity to bind into every reference",
    )
    parser.add_argument("--expected-policy-identity", required=True, type=Path)
    parser.add_argument("--policy-receipt", required=True, type=Path)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--readiness-endpoint", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _post_update_parser()
    args = parser.parse_args(argv)
    if len(args.input_record) != 8:
        parser.error("exactly eight --input-record values are required")
    try:
        build_post_update_references(
            endpoint=args.endpoint,
            input_records=args.input_record,
            output_path=args.output,
            base_model=args.base_model,
            policy_model=args.policy_model,
            policy=args.policy,
            expected_policy_identity=args.expected_policy_identity,
            policy_receipt=args.policy_receipt,
            cache_salt=args.cache_salt,
            readiness_endpoint=args.readiness_endpoint,
            timeout=args.timeout,
        )
    except PostUpdateQualificationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
