"""Compare eight repeated packed trainer records with one H200 reference.

This reports selected-token deltas, not the trainer's full parity guard.
Use the trainer guard result and the unchanged 600s limit for qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import build_trinity_synthetic_parity as synthetic
except ImportError:  # Direct script invocation on the qualification host.
    import build_trinity_synthetic_parity as synthetic


def _integer(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise synthetic.SyntheticParityError(f"{label} must be {expected}")


def _binding(record: Mapping[str, Any], reference: Mapping[str, Any], label: str) -> None:
    for field in ("input_ids", "position_ids", "sequence_lengths", "loss_mask"):
        value = record.get(field)
        expected = reference[field]
        if (
            not isinstance(value, list)
            or value != expected
            or any(type(a) is not type(b) for a, b in zip(value, expected))
        ):
            raise synthetic.SyntheticParityError(f"{label} {field} differs from H200")


def _trainer_values(
    record: Mapping[str, Any], reference: Mapping[str, Any], logs: list[float], label: str
) -> list[float]:
    _binding(record, reference, label)
    actual_reference = synthetic._numeric_array(
        record.get("reference_logprobs"), f"{label} reference_logprobs", synthetic.TOTAL_TOKENS
    )
    # FakeDataLoader casts H200 JSON floats to torch.float32 before the dump.
    expected = [struct.unpack("f", struct.pack("f", value))[0] for value in logs]
    if actual_reference != expected:
        raise synthetic.SyntheticParityError(f"{label} reference_logprobs differ from H200 float32")
    return synthetic._numeric_array(
        record.get("trainer_logprobs"), f"{label} trainer_logprobs", synthetic.TOTAL_TOKENS
    )


def _stats(left: list[float], right: list[float]) -> dict[str, float | int]:
    result = synthetic._stats(left, right, include_kl=True)
    if any(not math.isfinite(value) for value in result.values()):
        raise synthetic.SyntheticParityError("comparison metric is not finite")
    return result


def compare(
    trainer: Path | Mapping[str, Any],
    reference: Path | Mapping[str, Any],
    gpipe: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on missing records, invalid bindings, or missing logprobs."""
    reference_file = (
        {"path": str(reference), "size": reference.stat().st_size,
         "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
        if isinstance(reference, Path) else None
    )
    reference = synthetic._load_mapping(reference, "H200 packed reference")
    _integer(reference.get("schema_version"), 1, "H200 reference schema_version")
    synthetic.validate_source_kind(reference)
    positions = [position for length in synthetic.SEQUENCE_LENGTHS for position in range(length)]
    mask = [flag for length in synthetic.SEQUENCE_LENGTHS for flag in [False] + [True] * (length - 1)]
    ids = reference.get("input_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != synthetic.TOTAL_TOKENS
        or any(type(value) is not int or not 0 <= value < synthetic.VOCAB_SIZE for value in ids)
    ):
        raise synthetic.SyntheticParityError("H200 input_ids are invalid")
    _binding(reference, {
        "input_ids": ids,
        "position_ids": positions,
        "sequence_lengths": list(synthetic.SEQUENCE_LENGTHS),
        "loss_mask": mask,
    }, "H200 reference")
    synthetic._route_raw(reference.get("routed_experts"), synthetic.TOTAL_TOKENS)
    logs = synthetic._numeric_array(
        reference.get("inference_logprobs"), "H200 inference_logprobs", synthetic.TOTAL_TOKENS
    )
    trainer = synthetic._load_mapping(trainer, "packed 1f1b trainer")
    _integer(trainer.get("schema_version"), 2, "trainer schema_version")
    _integer(trainer.get("microbatch_count"), 8, "trainer microbatch_count")
    order = trainer.get("microbatch_order")
    if not isinstance(order, list) or order != list(range(8)) or any(type(i) is not int for i in order):
        raise synthetic.SyntheticParityError("trainer microbatch_order must be 0 through 7")
    records = trainer.get("microbatches")
    if not isinstance(records, list) or len(records) != 8:
        raise synthetic.SyntheticParityError("trainer must contain exactly 8 records")
    gpipe_values = None
    if gpipe is not None:
        gpipe = synthetic._load_mapping(gpipe, "packed gpipe trainer")
        if "microbatches" in gpipe:
            raise synthetic.SyntheticParityError("gpipe must be one packed record")
        gpipe_values = _trainer_values(gpipe, reference, logs, "gpipe")
    valid_logs = [value for value, flag in zip(logs, mask) if flag]
    valid_gpipe = (
        [value for value, flag in zip(gpipe_values, mask) if flag]
        if gpipe_values is not None else None
    )
    reports = []
    all_values = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise synthetic.SyntheticParityError(f"microbatch {index} must be an object")
        _integer(record.get("microbatch_index"), index, "microbatch_index")
        _integer(record.get("microbatch_count"), 8, "microbatch_count")
        values = _trainer_values(record, reference, logs, f"microbatch {index}")
        valid = [value for value, flag in zip(values, mask) if flag]
        all_values.extend(valid)
        report = {"microbatch_index": index, "trainer_vs_h200": _stats(valid, valid_logs)}
        sequences = []
        start = 0
        for sequence_index, length in enumerate(synthetic.SEQUENCE_LENGTHS):
            end = start + length
            sequence = {
                "sequence_index": sequence_index,
                "trainer_vs_h200": _stats(values[start + 1:end], logs[start + 1:end]),
            }
            if gpipe_values is not None:
                sequence["trainer_vs_gpipe"] = _stats(
                    values[start + 1:end], gpipe_values[start + 1:end]
                )
            sequences.append(sequence)
            start = end
        report["sequences"] = sequences
        if valid_gpipe is not None:
            report["trainer_vs_gpipe"] = _stats(valid, valid_gpipe)
        reports.append(report)
    totals = {"trainer_vs_h200": _stats(all_values, valid_logs * 8)}
    if valid_gpipe is not None:
        totals["trainer_vs_gpipe"] = _stats(all_values, valid_gpipe * 8)
    return {
        "schema_version": 1,
        "diagnostic_name": "trinity_synthetic_repeated_packed_1f1b",
        "microbatch_count": 8,
        "tokens_per_microbatch": synthetic.TOTAL_TOKENS,
        "total_positions": synthetic.TOTAL_TOKENS * 8,
        "total_retained": len(all_values),
        "sequence_lengths": list(synthetic.SEQUENCE_LENGTHS),
        "reference_sha256": synthetic.json_sha256(reference),
        "reference_file": reference_file,
        "trainer_sha256": synthetic.json_sha256(trainer),
        "gpipe_sha256": synthetic.json_sha256(gpipe) if gpipe is not None else None,
        "reference_binding": "H200 logprobs cast to float32 by FakeDataLoader",
        "metric_source": "selected loss_mask tokens; exp(trainer-reference)-delta-1",
        "parity_gates": dict(synthetic.PARITY_GATES),
        "qualification_pass": None,
        "qualification_note": "Check trainer parity guard and 600s forward/backward limit separately.",
        "microbatches": reports,
        "totals": totals,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--gpipe", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(args.trainer, args.reference, args.gpipe)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (synthetic.SyntheticParityError, OSError, OverflowError) as error:
        parser.exit(1, f"packed 1f1b comparison failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
