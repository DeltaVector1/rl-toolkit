from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from runner import label_names, request_json


def examples(prompt_directory: Path) -> list[dict]:
    rows = []
    for path in sorted(prompt_directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for row_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for field, expected_refusal in (("refusal", True), ("compliant", False)):
                    text = row.get(field)
                    if isinstance(text, str) and text.strip():
                        rows.append(
                            {
                                "source_file": path.name,
                                "source_row": row_number,
                                "field": field,
                                "expected_refusal": expected_refusal,
                                "text": text,
                            }
                        )
    return rows


def classify(example: dict, url: str, revision: str) -> dict:
    result = request_json(
        url.rstrip("/") + "/classify",
        {"texts": [example["text"]]},
        120,
        3,
    )
    record = {**example, "attempts": list(result.attempts)}
    if result.status != "completed" or result.value is None:
        return {**record, "status": result.status}
    try:
        if result.value["model_revision"] != revision:
            raise ValueError("classifier revision mismatch")
        item = result.value["results"][0]
        label = int(item["label"])
        if label not in label_names:
            raise ValueError("unknown label")
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {**record, "status": "invalid_classifier_response", "error": str(error)}
    return {
        **record,
        "status": "scored",
        "label": label,
        "label_name": label_names[label],
        "predicted_refusal": label_names[label] == "refusal",
        "confidence": float(item["confidence"]),
        "input_tokens": int(item["input_tokens"]),
        "truncated": bool(item["truncated"]),
    }


def summarize(records: list[dict]) -> dict:
    scored = [record for record in records if record["status"] == "scored"]
    tp = sum(record["expected_refusal"] and record["predicted_refusal"] for record in scored)
    fn = sum(record["expected_refusal"] and not record["predicted_refusal"] for record in scored)
    fp = sum(not record["expected_refusal"] and record["predicted_refusal"] for record in scored)
    tn = sum(not record["expected_refusal"] and not record["predicted_refusal"] for record in scored)
    return {
        "examples": len(records),
        "scored": len(scored),
        "unscored": len(records) - len(scored),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "refusal_recall": tp / (tp + fn) if tp + fn else None,
        "refusal_precision": tp / (tp + fp) if tp + fp else None,
        "non_refusal_recall": tn / (tn + fp) if tn + fp else None,
        "accuracy": (tp + tn) / len(scored) if scored else None,
        "truncated": sum(record.get("truncated", False) for record in scored),
    }


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def markdown_report(report: dict) -> str:
    return "\n".join(
        [
            "# Classifier check",
            "",
            f"Examples: {report['examples']:,}  ",
            f"Scored: {report['scored']:,}  ",
            f"Not scored: {report['unscored']:,}",
            "",
            "| Check | Result |",
            "|---|---:|",
            f"| Refusal precision | {percent(report['refusal_precision'])} |",
            f"| Refusal recall | {percent(report['refusal_recall'])} |",
            f"| Classifier input cut short | {report['truncated']:,} |",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--classifier-url", required=True)
    parser.add_argument(
        "--classifier-revision",
        default="2084304ece320e87820140adfc8845761fb357a0",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()
    source = examples(args.prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(
            pool.map(
                lambda row: classify(
                    row,
                    args.classifier_url,
                    args.classifier_revision,
                ),
                source,
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "responses-and-judgements.jsonl"
    with log_path.open("w", encoding="utf-8") as destination:
        for record in records:
            logged = {
                "dataset": record.get("source_file"),
                "source_row": record.get("source_row"),
                "expected": record.get("field"),
                "response": record.get("text"),
                "status": record.get("status"),
                "judgement": {
                    "label": record.get("label_name"),
                    "confidence": record.get("confidence"),
                    "refusal": record.get("predicted_refusal"),
                    "input_cut_short": record.get("truncated"),
                },
            }
            destination.write(json.dumps(logged, ensure_ascii=False) + "\n")
    report = summarize(records)
    report_path = args.output / "report.md"
    report_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Responses and judgements: {log_path}")


if __name__ == "__main__":
    main()
