from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


label_names = {0: "refusal", 1: "compliance"}
classifier_revision = "2084304ece320e87820140adfc8845761fb357a0"

source_names = {
    "MT.jsonl": "mt",
    "Nous-Refusals.jsonl": "nous_refusals",
    "eval.china-V1.jsonl": "china",
    "eval.dark-V1.jsonl": "dark",
    "eval.edge-V1.jsonl": "edge",
    "eval.hacking-V1.jsonl": "hacking",
    "venice.jsonl": "venice",
}

muse_user_channel = re.compile(
    r"to=user<\|message\|>(.*?)(?=<\|eot\|>|<\|eom\|>|$)",
    re.DOTALL,
)


def extract_muse_visible_text(text: str) -> str:
    return "\n".join(muse_user_channel.findall(text)).strip()


@dataclass(frozen=True)
class RequestResult:
    status: str
    value: dict | None
    attempts: tuple[dict, ...]


def request_json(
    url: str,
    body: dict,
    timeout_seconds: float,
    attempts: int,
) -> RequestResult:
    records = []
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                value = json.loads(response.read())
            records.append(
                {
                    "attempt": attempt,
                    "status": "completed",
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            return RequestResult("completed", value, tuple(records))
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as error:
            records.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            if attempt < attempts:
                time.sleep(0.25 * 2 ** (attempt - 1) + random.random() * 0.1)
    return RequestResult("unscored_service_failure", None, tuple(records))


def generate(
    index: int,
    row: dict,
    endpoints: list[str],
    model: str,
    max_tokens: int,
    timeout_seconds: float,
    attempts: int,
    raw_muse_output: bool,
) -> dict:
    endpoint = endpoints[index % len(endpoints)].rstrip("/")
    body = {
        "model": model,
        "messages": row["messages"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": max_tokens,
    }
    if raw_muse_output:
        body["skip_special_tokens"] = False
    result = request_json(
        endpoint + "/chat/completions",
        body,
        timeout_seconds,
        attempts,
    )
    record = {**row, "model_attempts": list(result.attempts)}
    if result.status != "completed" or result.value is None:
        return {**record, "evaluation_status": result.status}
    try:
        choice = result.value["choices"][0]
        text = str(choice["message"].get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as error:
        return {
            **record,
            "evaluation_status": "invalid_model_response",
            "error": str(error),
        }
    raw_response = text
    if raw_muse_output and muse_user_channel.search(text):
        text = extract_muse_visible_text(text)
    if not text:
        return {
            **record,
            "evaluation_status": "invalid_model_response",
            "error": "empty visible response",
            "raw_response": raw_response if raw_muse_output else None,
            "finish_reason": finish_reason,
        }
    return {
        **record,
        "evaluation_status": "generated",
        "response": text,
        "raw_response": raw_response if raw_muse_output else None,
        "finish_reason": finish_reason,
        "usage": result.value.get("usage", {}),
    }


def classify(
    record: dict,
    classifier_url: str,
    classifier_revision: str,
    timeout_seconds: float,
    attempts: int,
) -> dict:
    if record["evaluation_status"] != "generated":
        return record
    result = request_json(
        classifier_url.rstrip("/") + "/classify",
        {"texts": [record["response"]]},
        timeout_seconds,
        attempts,
    )
    record = {**record, "classifier_attempts": list(result.attempts)}
    if result.status != "completed" or result.value is None:
        return {**record, "evaluation_status": result.status}
    try:
        if result.value["model_revision"] != classifier_revision:
            raise ValueError("classifier revision mismatch")
        item = result.value["results"][0]
        label = int(item["label"])
        label_name = str(item["label_name"]).strip().lower()
        probabilities = [float(value) for value in item["probabilities"]]
        confidence = float(item["confidence"])
        if label not in label_names or len(probabilities) != 2:
            raise ValueError("invalid classifier label or probabilities")
        if label_name != label_names[label]:
            raise ValueError("classifier label name does not match its label")
        if abs(sum(probabilities) - 1.0) > 1e-3:
            raise ValueError("classifier probabilities do not sum to one")
        if abs(probabilities[label] - confidence) > 1e-4:
            raise ValueError("classifier confidence does not match its label")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {
            **record,
            "evaluation_status": "invalid_classifier_response",
            "error": str(error),
        }
    return {
        **record,
        "evaluation_status": "scored",
        "classifier_revision": classifier_revision,
        "classifier_label": label,
        "classifier_label_name": label_names[label],
        "classifier_confidence": confidence,
        "classifier_probabilities": probabilities,
        "classifier_input_tokens": int(item["input_tokens"]),
        "classifier_truncated": bool(item["truncated"]),
        "is_refusal": label_name == "refusal",
    }


def load_suite(path: Path, expected_sha256: str) -> tuple[list[dict], str]:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"suite SHA-256 mismatch: {actual_sha256}")
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("messages"), list) or not row["messages"]:
                    raise ValueError("suite row has no messages")
                rows.append(row)
    return rows, actual_sha256


def metric_summary(records: list[dict]) -> dict:
    statuses: dict[str, int] = {}
    for record in records:
        status = record["evaluation_status"]
        statuses[status] = statuses.get(status, 0) + 1
    scored = [record for record in records if record["evaluation_status"] == "scored"]
    total = len(scored)
    refused = sum(record["is_refusal"] for record in scored)
    return {
        "examples": len(records),
        "statuses": statuses,
        "scored_examples": total,
        "refusal": {
            "count": refused,
            "rate": refused / total if total else None,
        },
        "compliance": {
            "count": total - refused,
            "rate": (total - refused) / total if total else None,
        },
        "classifier_truncated": sum(
            record.get("classifier_truncated", False) for record in scored
        ),
    }


def record_source(record: dict) -> str:
    source_file = str(record.get("source_file") or "unknown")
    return str(record.get("source") or source_names.get(source_file, source_file))


def summary(records: list[dict], model: str, suite_sha256: str) -> dict:
    report = {
        "model": model,
        "suite_sha256": suite_sha256,
        **metric_summary(records),
    }
    observed_sources = sorted({record_source(record) for record in records})
    report["per_source"] = {
        source: metric_summary(
            [
                record
                for record in records
                if record_source(record) == source
            ]
        )
        for source in observed_sources
    }
    return report


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def markdown_report(report: dict) -> str:
    lines = [
        "# Refusal evaluation",
        "",
        f"Model: `{report['model']}`  ",
        f"Prompts: {report['examples']:,}  ",
        f"Scored: {report['scored_examples']:,}  ",
        f"Not scored: {report['examples'] - report['scored_examples']:,}",
        "",
        "## Overall",
        "",
        "| Result | Count | Rate |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Refusal", "refusal"),
        ("Compliance", "compliance"),
    ):
        value = report[key]
        lines.append(f"| {label} | {value['count']:,} | {percent(value['rate'])} |")
    lines.extend(
        [
            f"| Classifier input cut short | {report['classifier_truncated']:,} | "
            f"{percent(report['classifier_truncated'] / report['scored_examples'] if report['scored_examples'] else None)} |",
            "",
            "## By dataset",
            "",
            "| Dataset | Scored | Refusal | Compliance | Input cut short |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for source, value in report["per_source"].items():
        truncated_rate = (
            value["classifier_truncated"] / value["scored_examples"]
            if value["scored_examples"]
            else None
        )
        lines.append(
            f"| {source} | {value['scored_examples']:,} | "
            f"{percent(value['refusal']['rate'])} | "
            f"{percent(value['compliance']['rate'])} | {percent(truncated_rate)} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"Suite SHA-256: `{report['suite_sha256']}`  ",
            f"Elapsed time: {report.get('elapsed_seconds', 0.0):.1f} seconds",
            "",
        ]
    )
    return "\n".join(lines)


def logged_record(record: dict) -> dict:
    output = {
        "id": record.get("id"),
        "dataset": record_source(record),
        "source_file": record.get("source_file"),
        "source_row": record.get("source_row"),
        "category": record.get("category"),
        "prompt": record.get("messages"),
        "response": record.get("response"),
        "status": record.get("evaluation_status"),
    }
    if record.get("evaluation_status") == "scored":
        output["judgement"] = {
            "label": record.get("classifier_label_name"),
            "confidence": record.get("classifier_confidence"),
            "refusal": record.get("is_refusal"),
            "input_tokens": record.get("classifier_input_tokens"),
            "input_cut_short": record.get("classifier_truncated"),
        }
    elif record.get("error"):
        output["error"] = record["error"]
    return output


def write_outputs(output: Path, records: list[dict], report: dict) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "responses-and-judgements.jsonl"
    with log_path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(logged_record(record), ensure_ascii=False) + "\n")
    report_path = output / "report.md"
    report_path.write_text(markdown_report(report), encoding="utf-8")
    return report_path, log_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-sha256", default="")
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--display-name", default="")
    parser.add_argument("--classifier-url", required=True)
    parser.add_argument(
        "--classifier-revision",
        default=classifier_revision,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--classifier-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--raw-muse-output", action="store_true")
    args = parser.parse_args()
    rows, suite_sha256 = load_suite(args.suite, args.suite_sha256)
    if args.num_examples >= 0:
        rows = rows[: args.num_examples]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        generated = list(
            pool.map(
                lambda item: generate(
                    item[0],
                    item[1],
                    args.endpoint,
                    args.model,
                    args.max_tokens,
                    args.model_timeout_seconds,
                    args.attempts,
                    args.raw_muse_output,
                ),
                enumerate(rows),
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(
            pool.map(
                lambda record: classify(
                    record,
                    args.classifier_url,
                    args.classifier_revision,
                    args.classifier_timeout_seconds,
                    args.attempts,
                ),
                generated,
            )
        )
    report = summary(records, args.display_name or args.model, suite_sha256)
    report["elapsed_seconds"] = time.monotonic() - started
    report_path, log_path = write_outputs(args.output, records, report)
    print(f"Report: {report_path}")
    print(f"Responses and judgements: {log_path}")


if __name__ == "__main__":
    main()
