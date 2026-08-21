from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


whitespace_pattern = re.compile(r"\s+")
word_pattern = re.compile(r"[\w']+", re.UNICODE)
user_roles = {"human", "user"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return whitespace_pattern.sub(" ", text).strip()


def user_turns(row: dict) -> list[str]:
    conversation = (
        row.get("conversation")
        or row.get("conversations")
        or row.get("turns")
    )
    if isinstance(conversation, list):
        turns = []
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from") or turn.get("role") or "").casefold()
            text = str(turn.get("value") or turn.get("content") or "")
            if role in user_roles and normalize(text):
                turns.append(text)
        return turns
    prompt = row.get("prompt")
    if isinstance(prompt, str) and normalize(prompt):
        return [prompt]
    return []


def read_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def joined(turns: Iterable[str]) -> str:
    return "\n<next-user-turn>\n".join(normalize(turn) for turn in turns)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def shingles(text: str, size: int = 4) -> frozenset[int]:
    words = word_pattern.findall(normalize(text))
    if not words:
        return frozenset()
    if len(words) < size:
        values = [" ".join(words)]
    else:
        values = [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
    return frozenset(
        int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")
        for value in values
    )


def similarity(left: frozenset[int], right: frozenset[int]) -> tuple[float, float]:
    if not left or not right:
        return 0.0, 0.0
    shared = len(left & right)
    return shared / len(left | right), shared / min(len(left), len(right))


@dataclass(frozen=True)
class TrainingText:
    row_number: int
    kind: str
    text: str
    values: frozenset[int]


class TrainingIndex:
    def __init__(self) -> None:
        self.exact: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.texts: list[TrainingText] = []
        self.buckets: dict[int, list[int]] = defaultdict(list)

    def add(self, row_number: int, kind: str, text: str) -> None:
        text = normalize(text)
        if not text:
            return
        self.exact[digest(text)].append((row_number, kind))
        values = shingles(text)
        index = len(self.texts)
        self.texts.append(TrainingText(row_number, kind, text, values))
        for value in values:
            self.buckets[value].append(index)

    def exact_matches(self, text: str) -> list[tuple[int, str]]:
        text = normalize(text)
        return self.exact.get(digest(text), [])

    def nearest(self, text: str) -> tuple[TrainingText | None, float, float]:
        values = shingles(text)
        candidates: set[int] = set()
        shared_values = [value for value in values if value in self.buckets]
        shared_values.sort(key=lambda value: len(self.buckets[value]))
        for value in shared_values[:16]:
            candidates.update(self.buckets[value])
        best: TrainingText | None = None
        best_jaccard = 0.0
        best_containment = 0.0
        for index in candidates:
            entry = self.texts[index]
            jaccard, containment = similarity(values, entry.values)
            if (jaccard, containment) > (best_jaccard, best_containment):
                best = entry
                best_jaccard = jaccard
                best_containment = containment
        return best, best_jaccard, best_containment


def build_training_index(path: Path) -> TrainingIndex:
    index = TrainingIndex()
    for row_number, row in read_jsonl(path):
        turns = user_turns(row)
        for turn in turns:
            index.add(row_number, "user_turn", turn)
        if turns:
            index.add(row_number, "conversation", joined(turns))
    return index


def evaluate_row(
    index: TrainingIndex,
    turns: list[str],
    near_jaccard: float,
    near_containment: float,
) -> dict:
    full_text = joined(turns)
    final_text = turns[-1]
    exact_final = index.exact_matches(final_text)
    exact_conversation = index.exact_matches(full_text)
    exact_any_turn = []
    for turn_number, turn in enumerate(turns, 1):
        for training_row, kind in index.exact_matches(turn):
            exact_any_turn.append(
                {
                    "eval_turn": turn_number,
                    "training_row": training_row,
                    "training_kind": kind,
                }
            )
    nearest_final, final_jaccard, final_containment = index.nearest(final_text)
    nearest_full, full_jaccard, full_containment = index.nearest(full_text)
    final_shorter = min(
        len(shingles(final_text)),
        len(nearest_final.values) if nearest_final else 0,
    )
    full_shorter = min(
        len(shingles(full_text)),
        len(nearest_full.values) if nearest_full else 0,
    )
    near_final = (
        nearest_final is not None
        and not exact_final
        and final_shorter >= 8
        and (
            final_jaccard >= near_jaccard
            or (
                final_containment >= near_containment
                and final_jaccard >= 0.50
            )
        )
    )
    near_conversation = (
        nearest_full is not None
        and not exact_conversation
        and full_shorter >= 8
        and (
            full_jaccard >= near_jaccard
            or (
                full_containment >= near_containment
                and full_jaccard >= 0.50
            )
        )
    )
    exact_duplicate = bool(exact_final or exact_conversation)
    exact_overlap = bool(exact_any_turn or exact_conversation)
    return {
        "exact_duplicate": exact_duplicate,
        "exact_overlap": exact_overlap,
        "exact_final": [
            {"training_row": row, "training_kind": kind}
            for row, kind in exact_final
        ],
        "exact_conversation": [
            {"training_row": row, "training_kind": kind}
            for row, kind in exact_conversation
        ],
        "exact_any_turn": exact_any_turn,
        "near_duplicate": near_final or near_conversation,
        "near_final": {
            "matched": near_final,
            "training_row": nearest_final.row_number if nearest_final else None,
            "training_kind": nearest_final.kind if nearest_final else None,
            "jaccard": round(final_jaccard, 6),
            "containment": round(final_containment, 6),
            "shorter_shingle_count": final_shorter,
        },
        "near_conversation": {
            "matched": near_conversation,
            "training_row": nearest_full.row_number if nearest_full else None,
            "training_kind": nearest_full.kind if nearest_full else None,
            "jaccard": round(full_jaccard, 6),
            "containment": round(full_containment, 6),
            "shorter_shingle_count": full_shorter,
        },
    }


def analyze(
    training_path: Path,
    prompt_directory: Path,
    report_path: Path,
    filtered_directory: Path | None,
    near_jaccard: float,
    near_containment: float,
) -> dict:
    index = build_training_index(training_path)
    files = []
    totals = {
        "rows": 0,
        "recognized_rows": 0,
        "exact_duplicate_rows": 0,
        "exact_overlap_rows": 0,
        "near_duplicate_rows": 0,
        "rows_with_any_exact_turn": 0,
        "clean_rows": 0,
    }
    if filtered_directory:
        filtered_directory.mkdir(parents=True, exist_ok=True)
    details_path = report_path.with_suffix(".rows.jsonl")
    with details_path.open("w", encoding="utf-8") as details:
        for path in sorted(prompt_directory.glob("*.jsonl")):
            counts = {key: 0 for key in totals}
            filtered = (
                (filtered_directory / path.name).open("w", encoding="utf-8")
                if filtered_directory
                else None
            )
            try:
                for row_number, row in read_jsonl(path):
                    counts["rows"] += 1
                    turns = user_turns(row)
                    if not turns:
                        if filtered:
                            filtered.write(json.dumps(row, ensure_ascii=False) + "\n")
                        continue
                    counts["recognized_rows"] += 1
                    result = evaluate_row(
                        index,
                        turns,
                        near_jaccard,
                        near_containment,
                    )
                    if result["exact_duplicate"]:
                        counts["exact_duplicate_rows"] += 1
                    if result["exact_overlap"]:
                        counts["exact_overlap_rows"] += 1
                    if result["near_duplicate"]:
                        counts["near_duplicate_rows"] += 1
                    if result["exact_any_turn"]:
                        counts["rows_with_any_exact_turn"] += 1
                    contaminated = result["exact_overlap"] or result["near_duplicate"]
                    if not contaminated:
                        counts["clean_rows"] += 1
                        if filtered:
                            filtered.write(json.dumps(row, ensure_ascii=False) + "\n")
                    details.write(
                        json.dumps(
                            {
                                "eval_file": path.name,
                                "eval_row": row_number,
                                "eval_id": row.get("id") or row.get("conversation_id"),
                                "turn_count": len(turns),
                                **result,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            finally:
                if filtered:
                    filtered.close()
            files.append({"file": path.name, **counts})
            for key in totals:
                totals[key] += counts[key]
            print(
                f"{path.name}: {counts['clean_rows']:,} clean, "
                f"{counts['exact_overlap_rows']:,} exact overlaps, "
                f"{counts['near_duplicate_rows']:,} near matches",
                flush=True,
            )
    report = {
        "training_dataset": str(training_path),
        "training_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "prompt_directory": str(prompt_directory),
        "near_thresholds": {
            "four_word_shingle_jaccard": near_jaccard,
            "shorter_text_containment": near_containment,
        },
        "training_index": {
            "unique_normalized_texts": len(index.exact),
            "indexed_texts": len(index.texts),
        },
        "totals": totals,
        "all_recognized_eval_rows_are_exact_duplicates": (
            totals["recognized_rows"] > 0
            and totals["recognized_rows"] == totals["exact_duplicate_rows"]
        ),
        "files": files,
        "row_details": str(details_path),
        "filtered_directory": str(filtered_directory) if filtered_directory else None,
    }
    lines = [
        "# Training overlap check",
        "",
        f"Training file SHA-256: `{report['training_sha256']}`",
        "",
        "| Dataset | Rows | Exact overlap | Near match | Clean |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in files:
        lines.append(
            f"| {item['file']} | {item['recognized_rows']:,} | "
            f"{item['exact_overlap_rows']:,} | {item['near_duplicate_rows']:,} | "
            f"{item['clean_rows']:,} |"
        )
    lines.extend(
        [
            "",
            f"Clean prompts: {totals['clean_rows']:,}  ",
            f"Exact overlaps removed: {totals['exact_overlap_rows']:,}  ",
            f"Near matches removed: {totals['near_duplicate_rows']:,}",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--filtered-directory", type=Path)
    parser.add_argument("--near-jaccard", type=float, default=0.80)
    parser.add_argument("--near-containment", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        args.training,
        args.prompts,
        args.report,
        args.filtered_directory,
        args.near_jaccard,
        args.near_containment,
    )
    totals = report["totals"]
    print(f"Clean prompts: {totals['clean_rows']:,}")
    print(f"Exact overlaps removed: {totals['exact_overlap_rows']:,}")
    print(f"Near matches removed: {totals['near_duplicate_rows']:,}")


if __name__ == "__main__":
    main()
