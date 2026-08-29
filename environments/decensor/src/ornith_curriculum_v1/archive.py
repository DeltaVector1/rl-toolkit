from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


word_pattern = re.compile(r"[\w']+", re.UNICODE)


def normalize_task_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def task_content_id(family_id: str, category: str, task_text: str) -> str:
    content = {
        "category": category,
        "family_id": family_id,
        "task_text": normalize_task_text(task_text),
    }
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_similarity(first: str, second: str) -> float:
    first_words = set(word_pattern.findall(normalize_task_text(first)))
    second_words = set(word_pattern.findall(normalize_task_text(second)))
    if not first_words and not second_words:
        return 1.0
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


@dataclass(frozen=True)
class ArchiveRecord:
    task_id: str
    family_id: str
    category: str
    task_text: str
    proposer_policy_version: str
    proposer_adapter_hash: str
    solver_policy_version: str
    solver_adapter_hash: str
    verifier_id: str
    verifier_revision: str
    verifier_rubric_hash: str
    success_count: int
    valid_solution_count: int
    success_rate: float
    frontier_score: float
    novelty_score: float


class TaskArchive:
    def __init__(self, records: tuple[ArchiveRecord, ...] = ()) -> None:
        self._records = {record.task_id: record for record in records}

    @classmethod
    def load(cls, path: Path) -> "TaskArchive":
        if not path.exists():
            return cls()
        records: list[ArchiveRecord] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    records.append(ArchiveRecord(**json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid archive record at line {line_number}: {error}"
                    ) from error
        return cls(tuple(records))

    def records(self) -> tuple[ArchiveRecord, ...]:
        return tuple(self._records.values())

    def contains(self, task_id: str) -> bool:
        return task_id in self._records

    def nearest(self, task_text: str) -> tuple[str | None, float]:
        nearest_id: str | None = None
        nearest_similarity = 0.0
        for record in self._records.values():
            similarity = task_similarity(task_text, record.task_text)
            if similarity > nearest_similarity:
                nearest_id = record.task_id
                nearest_similarity = similarity
        return nearest_id, nearest_similarity

    def novelty(self, task_text: str) -> tuple[float, str | None, float]:
        nearest_id, similarity = self.nearest(task_text)
        return 1.0 - similarity, nearest_id, similarity

    def add(self, record: ArchiveRecord) -> bool:
        if record.task_id in self._records:
            return False
        expected_id = task_content_id(
            record.family_id,
            record.category,
            record.task_text,
        )
        if record.task_id != expected_id:
            raise ValueError("archive task ID does not match its content")
        self._records[record.task_id] = record
        return True

    def append(self, path: Path, record: ArchiveRecord) -> bool:
        if not self.add(record):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
        return True


@dataclass(frozen=True)
class StaticEvaluationSet:
    records: tuple[ArchiveRecord, ...]

    @classmethod
    def load(cls, path: Path) -> "StaticEvaluationSet":
        return cls(TaskArchive.load(path).records())

    def archive(self) -> TaskArchive:
        return TaskArchive(self.records)
