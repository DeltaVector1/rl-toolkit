from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


model_revision = "2084304ece320e87820140adfc8845761fb357a0"
expected_labels = {0: "refusal", 1: "compliance"}


@dataclass
class Work:
    texts: list[str]
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: Exception | None = None


class Classifier:
    def __init__(
        self,
        model_path: Path,
        device: str,
        batch_size: int,
        batch_wait_ms: int,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.batch_wait_seconds = batch_wait_ms / 1000
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.label_names = {
            int(label): str(name).strip().lower()
            for label, name in self.model.config.id2label.items()
        }
        if self.label_names != expected_labels:
            raise ValueError(
                f"classifier labels must be {expected_labels}, got {self.label_names}"
            )
        limits = [
            int(value)
            for value in (
                getattr(self.tokenizer, "model_max_length", 0),
                getattr(self.model.config, "max_position_embeddings", 0),
            )
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        if not limits:
            raise ValueError("classifier has no valid input token limit")
        self.max_input_tokens = min(limits)
        self.queue: queue.Queue[Work] = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, texts: list[str], timeout: float) -> dict[str, Any]:
        work = Work(texts=texts)
        self.queue.put(work)
        if not work.done.wait(timeout):
            raise TimeoutError("classification queue timed out")
        if work.error:
            raise work.error
        if work.result is None:
            raise RuntimeError("classification finished without a result")
        return work.result

    def _run(self) -> None:
        while True:
            first = self.queue.get()
            works = [first]
            text_count = len(first.texts)
            deadline = time.monotonic() + self.batch_wait_seconds
            while text_count < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    work = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                works.append(work)
                text_count += len(work.texts)
            try:
                self._classify_works(works)
            except Exception as error:
                for work in works:
                    work.error = error
                    work.done.set()

    def _classify_works(self, works: list[Work]) -> None:
        texts = [text for work in works for text in work.texts]
        token_counts = [
            len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            for text in texts
        ]
        results = []
        for start in range(0, len(texts), self.batch_size):
            part = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                part,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
                padding=True,
            ).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.float()
                probabilities = self.torch.softmax(logits, dim=-1).cpu()
            for offset, values in enumerate(probabilities):
                label = int(values.argmax().item())
                original_tokens = token_counts[start + offset]
                results.append(
                    {
                        "label": label,
                        "label_name": self.label_names[label],
                        "confidence": float(values[label].item()),
                        "probabilities": [float(value) for value in values.tolist()],
                        "input_tokens": original_tokens,
                        "truncated": original_tokens > self.max_input_tokens,
                    }
                )
        position = 0
        for work in works:
            count = len(work.texts)
            work.result = {
                "model_revision": model_revision,
                "max_input_tokens": self.max_input_tokens,
                "results": results[position : position + count],
            }
            position += count
            work.done.set()


class Handler(BaseHTTPRequestHandler):
    classifier: Classifier
    request_timeout_seconds: float

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        self._json(
            200,
            {
                "status": "ready",
                "model_revision": model_revision,
                "labels": self.classifier.label_names,
                "max_input_tokens": self.classifier.max_input_tokens,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/classify":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            texts = body.get("texts")
            if (
                not isinstance(texts, list)
                or not texts
                or len(texts) > self.classifier.batch_size
                or any(not isinstance(text, str) or not text.strip() for text in texts)
            ):
                raise ValueError("texts must contain one or more nonempty strings")
            result = self.classifier.submit(texts, self.request_timeout_seconds)
            self._json(200, result)
        except ValueError as error:
            self._json(400, {"error": str(error)})
        except Exception as error:
            self._json(503, {"error": f"{type(error).__name__}: {error}"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18150)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=int, default=20)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    Handler.classifier = Classifier(
        args.model,
        args.device,
        args.batch_size,
        args.batch_wait_ms,
    )
    Handler.request_timeout_seconds = args.request_timeout_seconds
    server = Server((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "status": "ready",
                "host": args.host,
                "port": args.port,
                "device": args.device,
                "model_revision": model_revision,
                "labels": Handler.classifier.label_names,
                "max_input_tokens": Handler.classifier.max_input_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
