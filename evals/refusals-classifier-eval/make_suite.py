from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


source_names = {
    "MT.jsonl": "mt",
    "Nous-Refusals.jsonl": "nous_refusals",
    "eval.china-V1.jsonl": "china",
    "eval.dark-V1.jsonl": "dark",
    "eval.edge-V1.jsonl": "edge",
    "eval.hacking-V1.jsonl": "hacking",
    "venice.jsonl": "venice",
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def messages_for_row(row: dict) -> list[dict[str, str]]:
    conversation = (
        row.get("conversation")
        or row.get("conversations")
        or row.get("turns")
    )
    if isinstance(conversation, list):
        messages = []
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from") or turn.get("role") or "").casefold()
            content = str(turn.get("value") or turn.get("content") or "").strip()
            if not content:
                continue
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant", "chatgpt", "bard"}:
                role = "assistant"
            elif role != "system":
                continue
            messages.append({"role": role, "content": content})
        while messages and messages[-1]["role"] != "user":
            messages.pop()
        return messages
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]
    return []


def allocate(capacities: dict[str, int], total: int) -> dict[str, int]:
    allocation = {name: 0 for name in capacities}
    active = [name for name in sorted(capacities) if capacities[name] > 0]
    while sum(allocation.values()) < total and active:
        for name in list(active):
            if sum(allocation.values()) >= total:
                break
            if allocation[name] >= capacities[name]:
                active.remove(name)
                continue
            allocation[name] += 1
    return allocation


def build_suite(
    prompt_directory: Path,
    overlap_rows: Path,
    output: Path,
    total: int | None,
    seed: int,
) -> dict:
    clean_rows: dict[str, set[int]] = defaultdict(set)
    for _, result in read_jsonl(overlap_rows):
        exact_overlap = result.get("exact_overlap")
        if exact_overlap is None:
            exact_overlap = bool(
                result.get("exact_any_turn") or result.get("exact_conversation")
            )
        if not exact_overlap and not result.get("near_duplicate"):
            clean_rows[result["eval_file"]].add(int(result["eval_row"]))

    candidates: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(prompt_directory.glob("*.jsonl")):
        allowed = clean_rows[path.name]
        for row_number, row in read_jsonl(path):
            if row_number not in allowed:
                continue
            messages = messages_for_row(row)
            if not messages:
                continue
            serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            source_hash = hashlib.sha256(serialized.encode()).hexdigest()
            candidates[path.name].append(
                {
                    "id": source_hash,
                    "source": source_names.get(path.name, path.stem.casefold()),
                    "source_file": path.name,
                    "source_row": row_number,
                    "source_id": row.get("id") or row.get("conversation_id"),
                    "category": str(
                        row.get("category")
                        or row.get("overarching_category")
                        or "unknown"
                    ),
                    "messages": messages,
                }
            )

    content_ids: set[str] = set()
    content_duplicate_rows = 0
    seen: set[str] = set()
    for name in sorted(candidates):
        for candidate in candidates[name]:
            if candidate["id"] in seen:
                content_duplicate_rows += 1
            else:
                seen.add(candidate["id"])
                content_ids.add(candidate["id"])

    capacities = {name: len(rows) for name, rows in candidates.items()}
    available = sum(capacities.values())
    if total is None:
        total = available
    if total > available:
        raise ValueError(f"requested {total} examples but only {available} are clean")
    allocation = allocate(capacities, total)
    rng = random.Random(seed)
    selected = []
    for name in sorted(candidates):
        rows = list(candidates[name])
        rng.shuffle(rows)
        selected.extend(rows[: allocation[name]])
    rng.shuffle(selected)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for row in selected:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    suite_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "suite": str(output),
        "suite_sha256": suite_sha256,
        "seed": seed,
        "requested_examples": total,
        "written_examples": len(selected),
        "clean_capacities": capacities,
        "unique_prompt_contents": len(content_ids),
        "repeated_prompt_rows": content_duplicate_rows,
        "allocation": allocation,
        "overlap_rows": str(overlap_rows),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--overlap-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", default="all")
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    report = build_suite(
        args.prompts,
        args.overlap_rows,
        args.output,
        None if args.examples == "all" else int(args.examples),
        args.seed,
    )
    print(f"Wrote {report['written_examples']:,} prompts to {args.output}")
    print(f"SHA-256: {report['suite_sha256']}")
    for source, count in sorted(report["allocation"].items()):
        print(f"{source}: {count:,}")


if __name__ == "__main__":
    main()
