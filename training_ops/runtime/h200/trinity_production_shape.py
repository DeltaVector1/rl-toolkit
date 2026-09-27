"""Fresh H200 references for seven production-shaped records and one control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path
import uuid

try:
    from . import build_trinity_synthetic_parity as synthetic
    from . import compare_trinity_packed_1f1b as dense
except ImportError:
    import build_trinity_synthetic_parity as synthetic
    import compare_trinity_packed_1f1b as dense


lengths = [8187, 7006, 6543, 6436, 4530]
seed = "trinity-production-shape-v1"


def requests(salt):
    if not isinstance(salt, str) or not salt:
        raise ValueError("fresh cache salt is required")
    return [
        dict(
            model=synthetic.DEFAULT_BASE_MODEL,
            token_ids=[synthetic._token(seed, index, pos) for pos in range(length)],
            sampling_params=dict(synthetic.SAMPLING_PARAMS),
            cache_salt=f"production-shape-{salt}-{index}",
        )
        for index, length in enumerate([*lengths, 32768])
    ]


def assemble(sent, responses):
    if len(sent) != 6 or len(responses) != 6:
        raise ValueError("six fresh H200 requests and responses are required")
    parsed = [
        synthetic._parse_response(response, request["token_ids"])
        for request, response in zip(sent, responses)
    ]
    tail = parsed[0][1]["shape"][1:]
    if any(route["shape"][1:] != tail for _, route, _ in parsed):
        raise ValueError("H200 route layer/top-k shapes differ")

    def record(indices, segment_lengths):
        return dict(
            source_kind="synthetic",
            input_ids=[token for i in indices for token in sent[i]["token_ids"]],
            inference_logprobs=[score for i in indices for score in parsed[i][0]],
            position_ids=[pos for length in segment_lengths for pos in range(length)],
            sequence_lengths=list(segment_lengths),
            loss_mask=[
                flag
                for length in segment_lengths
                for flag in [False] + [True] * (length - 1)
            ],
            routed_experts=synthetic._route_from_raw(
                b"".join(parsed[i][2] for i in indices), tail
            ),
        )

    production = record(range(5), lengths)
    control = record([5], [32768])
    return [copy.deepcopy(production) for _ in range(7)] + [control]


def build(*, endpoint, post_json=None):
    endpoint = synthetic._endpoint_url(endpoint)
    salt = uuid.uuid4().hex
    sent = requests(salt)
    post = post_json or synthetic._post_json
    responses = [post(endpoint, request, 120.0) for request in sent]
    records = assemble(sent, responses)
    return dict(
        schema_version=2,
        source_kind="synthetic",
        diagnostic_name="trinity_production_shape",
        base_model=synthetic.DEFAULT_BASE_MODEL,
        topology=dict(synthetic.TOPOLOGY),
        cache_salt=salt,
        requests=sent,
        responses=responses,
        request_sha256=[synthetic.json_sha256(item) for item in sent],
        response_sha256=[synthetic.json_sha256(item) for item in responses],
        record_sha256=[synthetic.json_sha256(item) for item in records],
        microbatches=records,
    )


def validate(reference):
    dense._integer(reference.get("schema_version"), 2, "reference schema")
    synthetic.validate_source_kind(reference)
    if (
        reference.get("base_model") != synthetic.DEFAULT_BASE_MODEL
        or reference.get("topology") != synthetic.TOPOLOGY
        or reference.get("diagnostic_name") != "trinity_production_shape"
    ):
        raise ValueError("reference model, topology or diagnostic identity differs")
    sent = requests(reference.get("cache_salt"))
    if synthetic.json_sha256(sent) != synthetic.json_sha256(reference.get("requests")):
        raise ValueError(
            "reference requests differ from fresh deterministic base requests"
        )
    responses = reference.get("responses")
    if not isinstance(responses, list):
        raise ValueError("reference responses are required")
    expected = assemble(sent, responses)
    for field, records in (
        ("request_sha256", sent),
        ("response_sha256", responses),
        ("record_sha256", expected),
    ):
        if reference.get(field) != [synthetic.json_sha256(item) for item in records]:
            raise ValueError(f"reference {field} differs")
    if synthetic.json_sha256(reference.get("microbatches")) != synthetic.json_sha256(
        expected
    ):
        raise ValueError("reference records differ from request/response evidence")


def padded(record):
    result = copy.deepcopy(record)
    count = 32768 - len(record["input_ids"])
    if count not in (0, 66):
        raise ValueError("unexpected padding count")
    for field, value in (
        ("input_ids", 1),
        ("position_ids", 0),
        ("inference_logprobs", 0.0),
        ("loss_mask", False),
    ):
        result[field].extend([value] * count)
    result["sequence_lengths"][-1] += count
    route, raw = synthetic._route_raw(
        record["routed_experts"], len(record["input_ids"])
    )
    tail = route["shape"][1:]
    result["routed_experts"] = synthetic._route_from_raw(
        raw + bytes(count * math.prod(tail)), tail
    )
    return result


def compare(trainer, reference):
    reference = synthetic._load_mapping(reference, "production reference")
    validate(reference)
    trainer = synthetic._load_mapping(trainer, "trainer")
    dense._integer(trainer.get("schema_version"), 2, "trainer schema")
    dense._integer(trainer.get("microbatch_count"), 8, "trainer count")
    order = trainer.get("microbatch_order")
    if order != list(range(8)) or any(type(i) is not int for i in order):
        raise ValueError("trainer order must be exactly 0 through 7")
    records = trainer.get("microbatches")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("trainer must contain eight records")
    reports, production_actual, production_expected = [], [], []
    for index, (actual, source) in enumerate(zip(records, reference["microbatches"])):
        if not isinstance(actual, dict):
            raise ValueError("trainer record must be an object")
        dense._integer(actual.get("microbatch_index"), index, "record index")
        dense._integer(actual.get("microbatch_count"), 8, "record count")
        expected = padded(source)
        logs = expected["inference_logprobs"]
        values = dense._trainer_values(actual, expected, logs, f"record {index}")
        mask = expected["loss_mask"]
        selected = [value for value, flag in zip(values, mask) if flag]
        selected_logs = [value for value, flag in zip(logs, mask) if flag]
        if len(selected) != (32697 if index < 7 else 32767):
            raise ValueError("selected-token count differs")
        sequences, start = [], 0
        for length in expected["sequence_lengths"]:
            indices = [i for i in range(start, start + length) if mask[i]]
            sequences.append(
                dense._stats([values[i] for i in indices], [logs[i] for i in indices])
            )
            start += length
        reports.append(
            dict(
                microbatch_index=index,
                sequences=sequences,
                trainer_vs_h200=dense._stats(selected, selected_logs),
            )
        )
        if index < 7:
            production_actual.extend(selected)
            production_expected.extend(selected_logs)
    return dict(
        schema_version=1,
        diagnostic_name="trinity_production_shape",
        production_selected=len(production_actual),
        control_selected=32767,
        production_total=dense._stats(production_actual, production_expected),
        control=reports[7],
        microbatches=reports,
        reference_sha256=synthetic.json_sha256(reference),
        trainer_sha256=synthetic.json_sha256(trainer),
        parity_gates=dict(synthetic.PARITY_GATES),
        qualification_pass=None,
        metric_source="Selected loss-mask positions; exp(trainer-H200)-delta-1; JSON H200 scores.",
        route_binding_limit="Reference routes bind H200 responses; trainer dumps omit routes. CPU loader/pad tests verify zero padding rows.",
        qualification_note="Diagnostic only. Check trainer gates and 600s forward/backward limit separately.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--endpoint", required=True)
    builder.add_argument("--output", type=Path, required=True)
    comparator = commands.add_parser("compare")
    comparator.add_argument("--trainer", type=Path, required=True)
    comparator.add_argument("--reference", type=Path, required=True)
    comparator.add_argument("--reference-sha256", required=True)
    comparator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output exists; refusing to clobber")
        if args.command == "build":
            result = build(endpoint=args.endpoint)
        else:
            if (
                hashlib.sha256(args.reference.read_bytes()).hexdigest()
                != args.reference_sha256
            ):
                raise ValueError("reference file SHA-256 differs")
            result = compare(args.trainer, args.reference)
            result["reference_file"] = {
                "sha256": args.reference_sha256,
                "size": args.reference.stat().st_size,
            }
        synthetic._write_json_bundle_no_clobber({args.output: result})
    except (ValueError, OSError, OverflowError) as error:
        parser.exit(1, f"production-shape diagnostic failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
