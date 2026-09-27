"""Re-score five preserved packed sequences on the loaded H200 policy.

This diagnostic sends the exact token IDs from ``parity-step1.json``.  It does
not tokenize, render prompts, read rollout batches, or emit response content.
The output contains hashes, identities, counts, and numeric comparisons only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

# Allow the supported direct file CLI as well as ``python -m`` execution.
if __package__ in {None, ""}:
    _repository_root = str(Path(__file__).resolve().parents[3])
    if _repository_root not in sys.path:
        sys.path.insert(0, _repository_root)

from training_ops.runtime.h200.build_trinity_parity_reference import (
    _extract_policy_receipt,
    _json_sha256,
    _normalise_route,
    _validate_policy_receipt,
)


SEQUENCE_COUNT = 5
DEFAULT_BASE_MODEL = "/data/trinity-thinking-fp8"
MATCH_ABS_TOLERANCE = 1e-4
MATCH_REL_TOLERANCE = 1e-4
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_RECEIPT_FIELDS = (
    "source_url",
    "target_sha256",
    "target_size",
    "config_sha256",
    "config_size",
    "lora_name",
    "server_instance",
)


class FixedSequenceRescoreError(ValueError):
    """The fixed-sequence diagnostic input, request, or evidence is unsafe."""


def _load_json(value: Path | Mapping[str, Any], label: str) -> dict[str, Any]:
    if isinstance(value, Path):
        if value.is_symlink() or not value.is_file():
            raise FixedSequenceRescoreError(f"{label} is missing or not a regular file")
        try:
            value = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FixedSequenceRescoreError(f"cannot read {label}") from error
    if not isinstance(value, Mapping):
        raise FixedSequenceRescoreError(f"{label} must be a JSON object")
    return dict(value)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest(), path.stat().st_size
    except OSError as error:
        raise FixedSequenceRescoreError(f"cannot hash parity record {path}") from error


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixedSequenceRescoreError(f"{label} is not numeric")
    value = float(value)
    if not math.isfinite(value) or value == -9999.0:
        raise FixedSequenceRescoreError(f"{label} is not finite evidence")
    return value


def _array(record: Mapping[str, Any], name: str, count: int) -> list[Any]:
    value = record.get(name)
    if not isinstance(value, list) or len(value) != count:
        raise FixedSequenceRescoreError(
            f"{name} must contain exactly {count} aligned values"
        )
    return list(value)


def load_parity_record(
    record: Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the preserved trainer parity artifact without changing it."""
    value = _load_json(record, "parity record")
    input_ids = value.get("input_ids")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(type(token) is not int or token < 0 for token in input_ids)
    ):
        raise FixedSequenceRescoreError("input_ids must be a non-empty non-negative integer list")
    lengths = value.get("sequence_lengths")
    if (
        not isinstance(lengths, list)
        or len(lengths) != SEQUENCE_COUNT
        or any(type(length) is not int or length <= 0 for length in lengths)
        or sum(lengths) != len(input_ids)
    ):
        raise FixedSequenceRescoreError(
            f"sequence_lengths must contain exactly {SEQUENCE_COUNT} positive lengths matching input_ids"
        )
    count = len(input_ids)
    positions = _array(value, "position_ids", count)
    if any(type(position) is not int or position < 0 for position in positions):
        raise FixedSequenceRescoreError("position_ids must contain non-negative integers")
    mask = _array(value, "loss_mask", count)
    if any(type(flag) is not bool for flag in mask) or not any(mask):
        raise FixedSequenceRescoreError("loss_mask must contain booleans and select tokens")
    trainer = [_finite(item, "trainer_logprobs") for item in _array(value, "trainer_logprobs", count)]
    reference = [_finite(item, "reference_logprobs") for item in _array(value, "reference_logprobs", count)]
    delta = [_finite(item, "delta") for item in _array(value, "delta", count)]
    if any(
        not math.isclose(actual, predicted - baseline, rel_tol=1e-6, abs_tol=1e-6)
        for actual, predicted, baseline in zip(delta, trainer, reference)
    ):
        raise FixedSequenceRescoreError("delta does not equal trainer_logprobs - reference_logprobs")
    expected_positions: list[int] = []
    for length in lengths:
        expected_positions.extend(range(length))
    position_mismatches = sum(
        actual != expected for actual, expected in zip(positions, expected_positions)
    )
    if isinstance(record, Path):
        reference_sha256, reference_size = _sha256_file(record)
    else:
        reference_sha256 = _json_sha256(value)
        reference_size = len(json.dumps(value, separators=(",", ":")).encode())
    return {
        "input_ids": [int(token) for token in input_ids],
        "position_ids": [int(position) for position in positions],
        "sequence_lengths": [int(length) for length in lengths],
        "loss_mask": list(mask),
        "trainer_logprobs": trainer,
        "reference_logprobs": reference,
        "delta": delta,
        "position_mismatches": position_mismatches,
        "reference_sha256": reference_sha256,
        "reference_size": reference_size,
    }


def _validate_endpoints(endpoint: str, readiness_endpoint: str) -> tuple[str, str]:
    if not isinstance(endpoint, str) or not endpoint:
        raise FixedSequenceRescoreError("generate endpoint is required")
    if not isinstance(readiness_endpoint, str) or not readiness_endpoint:
        raise FixedSequenceRescoreError("readiness endpoint is required")
    generated = urlsplit(endpoint.rstrip("/"))
    readiness = urlsplit(readiness_endpoint.rstrip("/"))
    if generated.scheme not in {"http", "https"} or not generated.netloc:
        raise FixedSequenceRescoreError("generate endpoint must be an HTTP URL")
    if not generated.path.endswith("/inference/v1/generate"):
        raise FixedSequenceRescoreError("generate endpoint must end with /inference/v1/generate")
    if (
        readiness.scheme != generated.scheme
        or readiness.netloc != generated.netloc
        or not readiness.path
        or readiness.path == generated.path
    ):
        raise FixedSequenceRescoreError(
            "readiness endpoint must use the generate endpoint origin and a distinct path"
        )
    return endpoint.rstrip("/"), readiness_endpoint.rstrip("/")


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
        raise FixedSequenceRescoreError(f"fixed sequence request failed: {error}") from error
    if not isinstance(value, dict):
        raise FixedSequenceRescoreError("fixed sequence response must be an object")
    return value


def _get_json(endpoint: str, timeout: float) -> dict[str, Any]:
    request = Request(endpoint, headers={"Connection": "close"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (OSError, HTTPError, URLError, json.JSONDecodeError) as error:
        raise FixedSequenceRescoreError(f"readiness request failed: {error}") from error
    if not isinstance(value, dict):
        raise FixedSequenceRescoreError("readiness response must be an object")
    return value


def _identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {field: receipt.get(field) for field in _RECEIPT_FIELDS}


def _validate_expected_identity(
    expected: Path | Mapping[str, Any], policy_model: str
) -> dict[str, Any]:
    value = _load_json(expected, "expected policy identity")
    try:
        validated = _validate_policy_receipt(value, value, policy_model)
    except (TypeError, ValueError) as error:
        raise FixedSequenceRescoreError(f"expected policy identity is invalid: {error}") from error
    if _identity(validated) != _identity(value):
        raise FixedSequenceRescoreError("expected policy identity is incomplete")
    return validated


def _receipt_event(
    get_json: Callable[[str, float], Mapping[str, Any]],
    readiness_endpoint: str,
    expected: Mapping[str, Any],
    policy_model: str,
    timeout: float,
    stage: str,
    request_index: int | None,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    payload = get_json(readiness_endpoint, timeout)
    if not isinstance(payload, Mapping):
        raise FixedSequenceRescoreError("readiness response is not an object")
    try:
        receipt = _validate_policy_receipt(
            _extract_policy_receipt(payload, policy_model), expected, policy_model
        )
    except (TypeError, ValueError) as error:
        raise FixedSequenceRescoreError(f"readiness receipt is invalid: {error}") from error
    if _identity(receipt) != _identity(baseline):
        raise FixedSequenceRescoreError("policy readiness receipt drifted")
    return {
        "stage": stage,
        "request_index": request_index,
        "identity": _identity(receipt),
    }


def _prompt_scores(response: Mapping[str, Any], token_ids: list[int]) -> list[float]:
    raw = response.get("prompt_logprobs")
    if not isinstance(raw, list) or len(raw) != len(token_ids):
        raise FixedSequenceRescoreError("prompt logprob count is not aligned")
    if raw[0] is not None:
        raise FixedSequenceRescoreError("first prompt logprob must be null")
    scores = [0.0]
    for index, token_id in enumerate(token_ids[1:], 1):
        entry = raw[index]
        if not isinstance(entry, Mapping):
            raise FixedSequenceRescoreError(f"prompt logprob {index} is missing")
        matches = [value for key, value in entry.items() if key == token_id or str(key) == str(token_id)]
        if len(matches) != 1 or not isinstance(matches[0], Mapping):
            raise FixedSequenceRescoreError(f"prompt logprob {index} token key is misaligned")
        scores.append(_finite(matches[0].get("logprob"), f"prompt logprob {index}"))
    return scores


def _choice(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise FixedSequenceRescoreError("response must contain exactly one choice")
    return choices[0]


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise FixedSequenceRescoreError("comparison has no selected tokens")
    absolute = [abs(value) for value in values]
    ordered = sorted(absolute)
    return {
        "count": len(values),
        "mean": math.fsum(values) / len(values),
        "mean_abs": math.fsum(absolute) / len(values),
        "p95_abs": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
        "max_abs": max(absolute),
    }


def _comparison(
    fresh: list[float],
    other: list[float],
    mask: list[bool],
) -> tuple[dict[str, Any], bool]:
    selected_deltas = [actual - expected for actual, expected, keep in zip(fresh, other, mask) if keep]
    all_deltas = [actual - expected for actual, expected in zip(fresh, other)]
    mismatches = sum(
        not math.isclose(actual, expected, rel_tol=MATCH_REL_TOLERANCE, abs_tol=MATCH_ABS_TOLERANCE)
        for actual, expected, keep in zip(fresh, other, mask)
        if keep
    )
    return {
        "selected": _stats(selected_deltas),
        "all": _stats(all_deltas),
        "selected_mismatch_count": mismatches,
        "selected_match": mismatches == 0,
        "abs_tolerance": MATCH_ABS_TOLERANCE,
        "rel_tolerance": MATCH_REL_TOLERANCE,
    }, mismatches == 0


def _sequence_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def _salt(prefix: str, index: int) -> str:
    if not isinstance(prefix, str) or not prefix.strip() or len(prefix) > 220:
        raise FixedSequenceRescoreError("fresh cache-salt prefix must be non-empty and at most 220 characters")
    salt = f"{prefix}-{index}"
    if len(salt) > 256:
        raise FixedSequenceRescoreError("fresh cache salt is too long")
    return salt


def _decision(reference_match: bool, trainer_match: bool) -> str:
    if reference_match and trainer_match:
        return "all_paths_match"
    if reference_match:
        return "fresh_matches_stored_rollout_but_trainer_differs"
    if trainer_match:
        return "fresh_matches_trainer_but_stored_rollout_differs"
    return "fresh_differs_from_stored_rollout"


def build_rescore(
    *,
    reference_path: Path | Mapping[str, Any],
    endpoint: str,
    readiness_endpoint: str,
    expected_policy_identity: Path | Mapping[str, Any],
    policy_model: str,
    fresh_salt_prefix: str,
    base_model: str = DEFAULT_BASE_MODEL,
    timeout: float = 120.0,
    post_json: Callable[[str, Mapping[str, Any], float], dict[str, Any]] = _post_json,
    get_json: Callable[[str, float], dict[str, Any]] = _get_json,
) -> dict[str, Any]:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise FixedSequenceRescoreError("timeout must be positive")
    if not isinstance(base_model, str) or not base_model.strip():
        raise FixedSequenceRescoreError("base model is required")
    if not isinstance(policy_model, str) or not policy_model.strip():
        raise FixedSequenceRescoreError("exact loaded policy LoRA model is required")
    endpoint, readiness_endpoint = _validate_endpoints(endpoint, readiness_endpoint)
    record = load_parity_record(reference_path)
    expected = _validate_expected_identity(expected_policy_identity, policy_model)
    baseline = _identity(expected)

    receipt_sequence: list[dict[str, Any]] = []
    receipt_sequence.append(
        _receipt_event(
            get_json,
            readiness_endpoint,
            expected,
            policy_model,
            float(timeout),
            "before_all",
            None,
            baseline,
        )
    )
    sequence_output: list[dict[str, Any]] = []
    reference_offset = 0
    requests_metadata: list[dict[str, Any]] = []
    fresh_flat: list[float] = []
    reference_flat: list[float] = []
    trainer_flat: list[float] = []
    mask_flat: list[bool] = []
    for index, length in enumerate(record["sequence_lengths"]):
        token_ids = record["input_ids"][reference_offset:reference_offset + length]
        mask = record["loss_mask"][reference_offset:reference_offset + length]
        stored_reference = record["reference_logprobs"][reference_offset:reference_offset + length]
        stored_trainer = record["trainer_logprobs"][reference_offset:reference_offset + length]
        positions = record["position_ids"][reference_offset:reference_offset + length]
        receipt_sequence.append(
            _receipt_event(
                get_json,
                readiness_endpoint,
                expected,
                policy_model,
                float(timeout),
                "before_request",
                index,
                baseline,
            )
        )
        request = {
            "model": policy_model,
            "token_ids": list(token_ids),
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "prompt_logprobs": 1,
            },
            "cache_salt": _salt(fresh_salt_prefix, index),
        }
        try:
            response = post_json(endpoint, request, float(timeout))
        except Exception as error:
            raise FixedSequenceRescoreError(f"sequence {index} request failed: {error}") from error
        if not isinstance(response, Mapping):
            raise FixedSequenceRescoreError(f"sequence {index} response is not an object")
        scores = _prompt_scores(response, token_ids)
        choice = _choice(response)
        raw_route = choice.get("routed_experts")
        if not isinstance(raw_route, Mapping):
            raise FixedSequenceRescoreError(f"sequence {index} response has no routed experts")
        try:
            route = _normalise_route(raw_route, len(token_ids))
        except (TypeError, ValueError) as error:
            raise FixedSequenceRescoreError(f"sequence {index} route is invalid: {error}") from error
        receipt_sequence.append(
            _receipt_event(
                get_json,
                readiness_endpoint,
                expected,
                policy_model,
                float(timeout),
                "after_request",
                index,
                baseline,
            )
        )
        fresh_flat.extend(scores)
        reference_flat.extend(stored_reference)
        trainer_flat.extend(stored_trainer)
        mask_flat.extend(mask)
        ref_comparison, _ = _comparison(scores, stored_reference, mask)
        trainer_comparison, _ = _comparison(scores, stored_trainer, mask)
        expected_position_mismatches = sum(
            actual != expected_position
            for actual, expected_position in zip(positions, range(length))
        )
        sequence_output.append(
            {
                "sequence_index": index,
                "sequence_sha256": _sequence_sha256(token_ids),
                "token_count": length,
                "selected_tokens": sum(mask),
                "masked_tokens": length - sum(mask),
                "position_mismatches": expected_position_mismatches,
                "route_shape": list(route["shape"]),
                "route_sha256": route["sha256"],
                "request_sha256": _json_sha256(request),
                "response_sha256": _json_sha256(response),
                "cache_salt": request["cache_salt"],
                "policy_identity": dict(baseline),
                "reference_identity": {
                    "artifact_sha256": record["reference_sha256"],
                    "artifact_size": record["reference_size"],
                    "sequence_index": index,
                    "sequence_sha256": _sequence_sha256(token_ids),
                    "reference_logprobs_sha256": _json_sha256(stored_reference),
                    "trainer_logprobs_sha256": _json_sha256(stored_trainer),
                },
                "comparisons": {
                    "fresh_vs_reference": ref_comparison,
                    "fresh_vs_trainer": trainer_comparison,
                },
            }
        )
        requests_metadata.append(
            {
                "sequence_index": index,
                "sequence_sha256": sequence_output[-1]["sequence_sha256"],
                "request_sha256": sequence_output[-1]["request_sha256"],
                "response_sha256": sequence_output[-1]["response_sha256"],
                "cache_salt": request["cache_salt"],
                "route_sha256": route["sha256"],
                "route_shape": list(route["shape"]),
                "policy_identity": dict(baseline),
                "reference_identity": sequence_output[-1]["reference_identity"],
            }
        )
        reference_offset += length
    receipt_sequence.append(
        _receipt_event(
            get_json,
            readiness_endpoint,
            expected,
            policy_model,
            float(timeout),
            "after_all",
            None,
            baseline,
        )
    )
    if reference_offset != len(record["input_ids"]):
        raise FixedSequenceRescoreError("fixed sequence offsets do not cover input_ids")
    position_mismatches = record["position_mismatches"]
    aggregate_reference_comparison, aggregate_reference_match = _comparison(
        fresh_flat, reference_flat, mask_flat
    )
    aggregate_trainer_comparison, aggregate_trainer_match = _comparison(
        fresh_flat, trainer_flat, mask_flat
    )
    decision = _decision(aggregate_reference_match, aggregate_trainer_match)
    return {
        "schema_version": 1,
        "diagnostic": "h200_fixed_sequence_policy0_rescore",
        "reference_sha256": record["reference_sha256"],
        "reference_size": record["reference_size"],
        "sequence_count": SEQUENCE_COUNT,
        "record_tokens": len(record["input_ids"]),
        "selected_tokens": sum(record["loss_mask"]),
        "masked_tokens": len(record["loss_mask"]) - sum(record["loss_mask"]),
        "position_mismatches": position_mismatches,
        "base_model": base_model,
        "policy_model": policy_model,
        "model_identity": {
            "base_model": base_model,
            "policy_model": policy_model,
        },
        "policy_identity": dict(baseline),
        "reference_identity": {
            "artifact_sha256": record["reference_sha256"],
            "artifact_size": record["reference_size"],
            "sequence_count": SEQUENCE_COUNT,
            "record_tokens": len(record["input_ids"]),
        },
        "receipt_sequence": receipt_sequence,
        "requests": requests_metadata,
        "sequences": sequence_output,
        "comparisons": {
            "fresh_vs_reference": aggregate_reference_comparison,
            "fresh_vs_trainer": aggregate_trainer_comparison,
        },
        "decision": decision,
        "match_tolerance": {
            "absolute": MATCH_ABS_TOLERANCE,
            "relative": MATCH_REL_TOLERANCE,
        },
    }


def publish_rescore(value: Mapping[str, Any], output_path: Path) -> None:
    """Write a diagnostic result without clobbering an existing artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output_path):
        raise FixedSequenceRescoreError(f"refusing to clobber existing output: {output_path}")
    temporary = output_path.with_name(
        f"{output_path.name}.partial.fixed-sequence.{os.getpid()}"
    )
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output_path, follow_symlinks=False)
        directory_fd = os.open(output_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise FixedSequenceRescoreError("refusing to clobber fixed-sequence output") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_rescore(*, output_path: Path, **kwargs: Any) -> dict[str, Any]:
    output_path = Path(output_path)
    if os.path.lexists(output_path):
        raise FixedSequenceRescoreError(f"refusing to clobber existing output: {output_path}")
    result = build_rescore(**kwargs)
    publish_rescore(result, output_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", "--reference-path", dest="reference", type=Path, required=True)
    parser.add_argument("--endpoint", "--generate-endpoint", dest="endpoint", required=True)
    parser.add_argument("--readiness-endpoint", required=True)
    parser.add_argument("--expected-policy-identity", type=Path, required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--fresh-salt-prefix", "--cache-salt-prefix", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", "--output-path", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_rescore(
            reference_path=args.reference,
            endpoint=args.endpoint,
            readiness_endpoint=args.readiness_endpoint,
            expected_policy_identity=args.expected_policy_identity,
            policy_model=args.policy_model,
            fresh_salt_prefix=args.fresh_salt_prefix,
            base_model=args.base_model,
            output_path=args.output,
            timeout=args.timeout,
        )
    except (OSError, FixedSequenceRescoreError) as error:
        print(f"fixed sequence rescore failed: {error}", file=sys.stderr)
        return 2
    print(f"fixed sequence rescore written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
