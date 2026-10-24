"""Derive and compare the fixed production-density synthetic mask diagnostic.

No network access is used. Qualification needs the separate trainer guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from . import build_trinity_synthetic_parity as synthetic
    from . import compare_trinity_packed_1f1b as dense
except ImportError:
    import build_trinity_synthetic_parity as synthetic
    import compare_trinity_packed_1f1b as dense


def mask_spec() -> dict[str, Any]:
    return {
        "prefix_false_counts": [410, 563, 666, 768, 870],
        "last_sequence_tail_false_count": 71,
        "selected": 29420,
        "excluded": 3348,
    }


def loss_mask() -> list[bool]:
    spec = mask_spec()
    mask = []
    for index, (length, prefix) in enumerate(
        zip(synthetic.SEQUENCE_LENGTHS, spec["prefix_false_counts"])
    ):
        tail = spec["last_sequence_tail_false_count"] if index == 4 else 0
        if prefix + tail >= length:
            raise synthetic.SyntheticParityError("mask prefix and tail overlap")
        mask.extend(
            [False] * prefix + [True] * (length - prefix - tail) + [False] * tail
        )
    if sum(mask) != 29420 or len(mask) - sum(mask) != 3348:
        raise synthetic.SyntheticParityError("mask counts differ")
    return mask


def validate_source(source: Mapping[str, Any]) -> None:
    """Use the existing synthetic field, route, and numeric validators."""
    dense._integer(source.get("schema_version"), 1, "source schema_version")
    synthetic.validate_source_kind(source)
    if "mask_derivation" in source or "microbatches" in source:
        raise synthetic.SyntheticParityError(
            "source must be the original packed reference"
        )
    tokens = [
        token for sequence in synthetic.build_token_sequences() for token in sequence
    ]
    dense._binding(
        source,
        {
            "input_ids": tokens,
            "position_ids": [i for n in synthetic.SEQUENCE_LENGTHS for i in range(n)],
            "sequence_lengths": list(synthetic.SEQUENCE_LENGTHS),
            "loss_mask": [
                flag
                for n in synthetic.SEQUENCE_LENGTHS
                for flag in [False] + [True] * (n - 1)
            ],
        },
        "source",
    )
    synthetic._route_raw(source.get("routed_experts"), synthetic.TOTAL_TOKENS)
    synthetic._numeric_array(
        source.get("inference_logprobs"), "source logprobs", synthetic.TOTAL_TOKENS
    )


def derive(source: Path, expected_sha256: str, expected_size: int) -> dict[str, Any]:
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        digest != expected_sha256
        or type(expected_size) is not int
        or len(raw) != expected_size
    ):
        raise synthetic.SyntheticParityError("source file SHA-256 or size differs")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise synthetic.SyntheticParityError("source must be an object")
    validate_source(value)
    value["loss_mask"] = loss_mask()
    value["mask_derivation"] = {
        "schema_version": 1,
        "source_file_sha256": digest,
        "source_file_size": len(raw),
        "mask_spec": mask_spec(),
        "limitation": "Production density only; synthetic sequence lengths and mask locations differ.",
    }
    return value


def validate_reference(reference: Mapping[str, Any], source: Path) -> None:
    metadata = reference.get("mask_derivation")
    if not isinstance(metadata, dict):
        raise synthetic.SyntheticParityError("mask derivation metadata is required")
    expected = derive(
        source, metadata.get("source_file_sha256"), metadata.get("source_file_size")
    )
    # Canonical JSON equality also rejects bool/int substitutions and extra fields.
    if synthetic.json_sha256(reference) != synthetic.json_sha256(expected):
        raise synthetic.SyntheticParityError(
            "derived reference differs from source or fixed mask spec"
        )


def records(
    artifact: Mapping[str, Any], reference: Mapping[str, Any], label: str
) -> list[list[float]]:
    dense._integer(artifact.get("schema_version"), 2, f"{label} schema_version")
    dense._integer(artifact.get("microbatch_count"), 8, f"{label} microbatch_count")
    order = artifact.get("microbatch_order")
    if order != list(range(8)) or any(type(i) is not int for i in order):
        raise synthetic.SyntheticParityError(f"{label} microbatch order differs")
    batches = artifact.get("microbatches")
    if not isinstance(batches, list) or len(batches) != 8:
        raise synthetic.SyntheticParityError(f"{label} requires eight records")
    values = []
    for index, record in enumerate(batches):
        if not isinstance(record, Mapping):
            raise synthetic.SyntheticParityError(f"{label} record must be an object")
        dense._integer(record.get("microbatch_index"), index, "microbatch_index")
        dense._integer(record.get("microbatch_count"), 8, "microbatch_count")
        values.append(
            dense._trainer_values(
                record, reference, reference["inference_logprobs"], label
            )
        )
    return values


def compare(
    trainer: Path | Mapping[str, Any],
    reference: Path | Mapping[str, Any],
    source: Path,
    dense_trainer: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reference_path = reference if isinstance(reference, Path) else None
    reference = synthetic._load_mapping(reference, "masked reference")
    validate_reference(reference, source)
    trainer = synthetic._load_mapping(trainer, "masked trainer")
    values = records(trainer, reference, "masked trainer")
    baseline = None
    baseline_artifact = None
    if dense_trainer is not None:
        baseline_artifact = synthetic._load_mapping(dense_trainer, "dense trainer")
        original = synthetic._load_mapping(source, "source")
        baseline = records(baseline_artifact, original, "dense trainer")
    mask = reference["loss_mask"]
    logs = reference["inference_logprobs"]

    def selected(
        items: list[float], start: int = 0, end: int = synthetic.TOTAL_TOKENS
    ) -> list[float]:
        return [items[i] for i in range(start, end) if mask[i]]

    reports = []
    for index, actual in enumerate(values):
        report = {
            "microbatch_index": index,
            "trainer_vs_h200": dense._stats(selected(actual), selected(logs)),
        }
        sequences = []
        start = 0
        for sequence_index, length in enumerate(synthetic.SEQUENCE_LENGTHS):
            end = start + length
            sequence = {
                "sequence_index": sequence_index,
                "trainer_vs_h200": dense._stats(
                    selected(actual, start, end), selected(logs, start, end)
                ),
            }
            if baseline is not None:
                sequence["trainer_vs_dense"] = dense._stats(
                    selected(actual, start, end), selected(baseline[index], start, end)
                )
            sequences.append(sequence)
            start = end
        report["sequences"] = sequences
        if baseline is not None:
            report["trainer_vs_dense"] = dense._stats(
                selected(actual), selected(baseline[index])
            )
        reports.append(report)
    all_values = [item for actual in values for item in selected(actual)]
    totals = {"trainer_vs_h200": dense._stats(all_values, selected(logs) * 8)}
    if baseline is not None:
        totals["trainer_vs_dense"] = dense._stats(
            all_values, [item for actual in baseline for item in selected(actual)]
        )
    return {
        "schema_version": 1,
        "diagnostic_name": "trinity_synthetic_masked_compaction",
        "microbatch_count": 8,
        "tokens_per_microbatch": synthetic.TOTAL_TOKENS,
        "selected_per_microbatch": sum(mask),
        "total_positions": synthetic.TOTAL_TOKENS * 8,
        "total_retained": len(all_values),
        "mask_derivation": reference["mask_derivation"],
        "reference_sha256": synthetic.json_sha256(reference),
        "reference_file": (
            {
                "size": reference_path.stat().st_size,
                "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            }
            if reference_path is not None
            else None
        ),
        "trainer_sha256": synthetic.json_sha256(trainer),
        "dense_trainer_sha256": synthetic.json_sha256(baseline_artifact)
        if baseline_artifact is not None
        else None,
        "reference_binding": "Every record binds tokens, positions, lengths, mask and float32 reference scores.",
        "route_binding_limit": "Routes bind source to derived input; trainer dumps do not contain routes.",
        "metric_source": "Selected loss_mask tokens; exp(trainer-reference)-delta-1; H200 JSON scores.",
        "parity_gates": dict(synthetic.PARITY_GATES),
        "qualification_pass": None,
        "qualification_note": "Check trainer parity guard and 600s forward/backward limit separately.",
        "microbatches": reports,
        "totals": totals,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("derive")
    builder.add_argument("--source", required=True, type=Path)
    builder.add_argument("--source-sha256", required=True)
    builder.add_argument("--source-size", required=True, type=int)
    builder.add_argument("--output", required=True, type=Path)
    comparator = commands.add_parser("compare")
    comparator.add_argument("--source", required=True, type=Path)
    comparator.add_argument("--reference", required=True, type=Path)
    comparator.add_argument("--trainer", required=True, type=Path)
    comparator.add_argument("--dense-trainer", type=Path)
    comparator.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise synthetic.SyntheticParityError("output exists; refusing to clobber")
        result = (
            derive(args.source, args.source_sha256, args.source_size)
            if args.command == "derive"
            else compare(args.trainer, args.reference, args.source, args.dense_trainer)
        )
        synthetic._write_json_bundle_no_clobber({args.output: result})
    except (ValueError, OSError, OverflowError) as error:
        parser.exit(1, f"masked compaction diagnostic failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
