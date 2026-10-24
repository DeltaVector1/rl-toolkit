from __future__ import annotations

from typing import Any, Literal

from datasets import load_dataset
import verifiers.v1 as vf

from refusals_classifier_eval_v1.client import classify_response


refusals_classifier_dataset = "NewEden/refusals-classifier-eval"


class RefusalsClassifierEvalData(vf.TaskData):
    source: str
    source_file: str
    source_row: int
    source_id: str | None = None
    category: str = "unknown"


class RefusalsClassifierEvalTaskConfig(vf.TaskConfig):
    classifier_base_url: str = "http://127.0.0.1:18250"
    classifier_revision: str = "2084304ece320e87820140adfc8845761fb357a0"
    attempts: int = 3
    request_timeout_seconds: float = 60.0
    total_timeout_seconds: float = 150.0


class RefusalsClassifierEvalTasksetConfig(vf.TasksetConfig):
    evaluation_set: Literal["training", "final"] = "training"
    num_examples: int = -1
    task: RefusalsClassifierEvalTaskConfig = RefusalsClassifierEvalTaskConfig()


def _last_assistant_text(trace: vf.Trace) -> str:
    messages = trace.branches[-1].messages if trace.branches else []
    for message in reversed(messages):
        if message.role == "assistant":
            return str(message.content or "").strip()
    return ""


class RefusalsClassifierEvalTask(
    vf.Task[
        RefusalsClassifierEvalData,
        vf.State,
        RefusalsClassifierEvalTaskConfig,
    ]
):
    @vf.metric
    async def refusal_classifier(self, trace: vf.Trace) -> dict[str, float]:
        text = _last_assistant_text(trace)
        if not text:
            trace.info["refusals_classifier_eval"] = {
                "evaluation_status": "invalid_model_response",
                "reason": "empty_visible_response",
                "source": self.data.source,
                "source_file": self.data.source_file,
                "source_row": self.data.source_row,
            }
            return {"refusal_classifier_invalid_response": 1.0}
        try:
            result, attempts = await classify_response(
                self.config.classifier_base_url,
                text,
                self.config.attempts,
                self.config.request_timeout_seconds,
                self.config.total_timeout_seconds,
            )
        except RuntimeError as error:
            trace.info["refusals_classifier_eval"] = {
                "evaluation_status": "unscored_service_failure",
                "error": str(error),
                "source": self.data.source,
                "source_file": self.data.source_file,
                "source_row": self.data.source_row,
            }
            raise
        if result.model_revision != self.config.classifier_revision:
            raise RuntimeError(
                "refusal classifier revision does not match the configured revision"
            )
        refusal = result.label_name == "refusal"
        trace.info["refusals_classifier_eval"] = {
            "evaluation_status": "scored",
            "source_file": self.data.source_file,
            "source": self.data.source,
            "source_row": self.data.source_row,
            "source_id": self.data.source_id,
            "category": self.data.category,
            "classifier_revision": result.model_revision,
            "classifier_label": result.label,
            "classifier_label_name": result.label_name,
            "classifier_confidence": result.confidence,
            "classifier_probabilities": list(result.probabilities),
            "classifier_input_tokens": result.input_tokens,
            "classifier_truncated": result.truncated,
            "attempts": attempts,
        }
        metrics = {
            "refusal_classifier_refusal": float(refusal),
            "refusal_classifier_non_refusal": float(not refusal),
            "refusal_classifier_compliance": float(not refusal),
            "refusal_classifier_confidence": result.confidence,
            "refusal_classifier_input_tokens": float(result.input_tokens),
            "refusal_classifier_input_truncated": float(result.truncated),
            "refusal_classifier_invalid_response": 0.0,
        }
        source_prefix = f"refusal_classifier/source/{self.data.source}"
        metrics |= {
            f"{source_prefix}/refusal": float(refusal),
            f"{source_prefix}/compliance": float(not refusal),
            f"{source_prefix}/input_truncated": float(result.truncated),
        }
        return metrics


class RefusalsClassifierEvalTaskset(
    vf.Taskset[RefusalsClassifierEvalTask, RefusalsClassifierEvalTasksetConfig]
):
    def load(self) -> list[RefusalsClassifierEvalTask]:
        data_file = f"{self.config.evaluation_set}-eval-set.jsonl"
        dataset = load_dataset(
            refusals_classifier_dataset,
            data_files={"train": data_file},
            split="train",
        )
        rows = [dict(row) for row in dataset]
        if self.config.num_examples >= 0:
            rows = rows[: self.config.num_examples]
        tasks = []
        for idx, row in enumerate(rows):
            messages = row["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"evaluation row {idx} has no messages")
            if not all(isinstance(message, dict) for message in messages):
                raise ValueError(f"evaluation row {idx} has invalid messages")
            source = row["source"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"evaluation row {idx} has no source")
            source_file = str(row.get("source_file") or data_file)
            source_row_value = row.get("source_row")
            source_row = (
                int(source_row_value)
                if source_row_value is not None
                else idx + 1
            )
            tasks.append(
                RefusalsClassifierEvalTask(
                    RefusalsClassifierEvalData(
                        idx=idx,
                        prompt=messages,
                        source=source.strip(),
                        source_file=source_file,
                        source_row=source_row,
                        source_id=row.get("source_id"),
                        category=str(row.get("category") or "unknown"),
                    ),
                    self.config.task,
                )
            )
        return tasks


__all__ = ["RefusalsClassifierEvalTaskset"]
